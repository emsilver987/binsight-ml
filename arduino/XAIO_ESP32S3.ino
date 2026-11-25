/*
 * BinSight capture firmware — XIAO ESP32-S3 Sense
 *
 * Captures short bursts of QVGA JPEG frames to the microSD card for dataset
 * collection. Driven entirely by Serial Monitor commands (115200 baud,
 * line ending set to "Newline").
 *
 * Commands:
 *   item 0001   set the current item ID (creates /item0001/ on the card)
 *   cap         capture one burst of FRAMES_PER_BURST frames for the current item
 *   undo        delete the last burst you captured (bad framing, hand in the way...)
 *   ls          list the files for the current item
 *   status      show current item, next burst number, free space
 *   help        show this list
 *
 * File layout:  /item0001/item0001_b0_00.jpg ... item0001_b2_04.jpg
 *   b<N> = burst number (matches the `burst` column in capture_log.csv)
 *   last two digits = frame within the burst
 *
 * Board settings (Tools menu): Board = XIAO_ESP32S3, PSRAM = OPI PSRAM
 */

 #include <Arduino.h>
 #include "esp_camera.h"
 #include "FS.h"
 #include "SD.h"
 #include "SPI.h"
 
 // ---------------------------------------------------------------------------
 // Capture settings — change these, not the code below
 // ---------------------------------------------------------------------------
 const int FRAMES_PER_BURST = 5;     // 3 bursts x 5 frames = 15 frames per item
 const int FRAME_INTERVAL_MS = 200;  // 5 fps
 const int WARMUP_FRAMES = 4;        // discarded before each burst (see captureBurst)
 const int JPEG_QUALITY = 10;        // 0-63, lower = better quality, bigger files
 const int SD_CS_PIN = 21;           // XIAO Sense microSD chip select
 
 // ---------------------------------------------------------------------------
 // XIAO ESP32-S3 Sense camera pins, hardcoded.
 // We skip board_config.h / camera_pins.h on purpose: this firmware only ever
 // targets one board, and hardcoding removes the "two models defined" failure
 // that crashed CameraWebServer.
 // Note: no LED/flash setup. GPIO 21 is the onboard LED *and* the SD chip
 // select, so touching the LED would corrupt SD writes.
 // ---------------------------------------------------------------------------
 #define PWDN_GPIO_NUM  -1
 #define RESET_GPIO_NUM -1
 #define XCLK_GPIO_NUM  10
 #define SIOD_GPIO_NUM  40
 #define SIOC_GPIO_NUM  39
 #define Y9_GPIO_NUM    48
 #define Y8_GPIO_NUM    11
 #define Y7_GPIO_NUM    12
 #define Y6_GPIO_NUM    14
 #define Y5_GPIO_NUM    16
 #define Y4_GPIO_NUM    18
 #define Y3_GPIO_NUM    17
 #define Y2_GPIO_NUM    15
 #define VSYNC_GPIO_NUM 38
 #define HREF_GPIO_NUM  47
 #define PCLK_GPIO_NUM  13
 
 // ---------------------------------------------------------------------------
 // State
 // ---------------------------------------------------------------------------
 String currentItem = "";   // e.g. "0001"; empty until `item` is used
 int lastBurst = -1;        // burst number most recently captured (for undo)
 String lineBuf = "";       // accumulates serial characters until newline
 
 // ---------------------------------------------------------------------------
 // Camera
 // ---------------------------------------------------------------------------
 bool initCamera() {
   camera_config_t config = {};
   config.ledc_channel = LEDC_CHANNEL_0;
   config.ledc_timer = LEDC_TIMER_0;
   config.pin_d0 = Y2_GPIO_NUM;
   config.pin_d1 = Y3_GPIO_NUM;
   config.pin_d2 = Y4_GPIO_NUM;
   config.pin_d3 = Y5_GPIO_NUM;
   config.pin_d4 = Y6_GPIO_NUM;
   config.pin_d5 = Y7_GPIO_NUM;
   config.pin_d6 = Y8_GPIO_NUM;
   config.pin_d7 = Y9_GPIO_NUM;
   config.pin_xclk = XCLK_GPIO_NUM;
   config.pin_pclk = PCLK_GPIO_NUM;
   config.pin_vsync = VSYNC_GPIO_NUM;
   config.pin_href = HREF_GPIO_NUM;
   config.pin_sccb_sda = SIOD_GPIO_NUM;
   config.pin_sccb_scl = SIOC_GPIO_NUM;
   config.pin_pwdn = PWDN_GPIO_NUM;
   config.pin_reset = RESET_GPIO_NUM;
   config.xclk_freq_hz = 20000000;
 
   // Allocate buffers at QVGA directly (CameraWebServer allocated at UXGA so the
   // browser could switch resolutions). Smaller buffers leave PSRAM for the
   // model later.
   config.frame_size = FRAMESIZE_QVGA;      // 320x240, matches the dataset spec
   config.pixel_format = PIXFORMAT_JPEG;    // sensor compresses; we write bytes as-is
   config.jpeg_quality = JPEG_QUALITY;
   config.fb_location = CAMERA_FB_IN_PSRAM;
 
   // Two buffers + GRAB_LATEST: the driver keeps filling the spare buffer in the
   // background and hands us the newest frame, so timestamps reflect "now"
   // rather than a frame captured whenever we last asked.
   config.fb_count = 2;
   config.grab_mode = CAMERA_GRAB_LATEST;
 
   esp_err_t err = esp_camera_init(&config);
   if (err != ESP_OK) {
     Serial.printf("Camera init failed: 0x%x\n", err);
     return false;
   }
   return true;
 }
 
 // ---------------------------------------------------------------------------
 // SD helpers
 // ---------------------------------------------------------------------------
 String itemDir() { return "/item" + currentItem; }
 
 String framePath(int burst, int frame) {
   char name[48];
   snprintf(name, sizeof(name), "/item%s/item%s_b%d_%02d.jpg",
            currentItem.c_str(), currentItem.c_str(), burst, frame);
   return String(name);
 }
 
 // Next burst number = first N whose frame 00 doesn't exist yet.
 // Scanning the card (instead of keeping a counter in RAM) means a reboot or a
 // return to an earlier item never overwrites existing bursts.
 int nextBurstNumber() {
   int b = 0;
   while (SD.exists(framePath(b, 0))) b++;
   return b;
 }
 
 // ---------------------------------------------------------------------------
 // Capture
 // ---------------------------------------------------------------------------
 void captureBurst() {
   if (currentItem == "") {
     Serial.println("ERR: set an item first, e.g. `item 0001`");
     return;
   }
   int burst = nextBurstNumber();
 
   // Warm-up: the sensor runs auto-exposure and auto white balance continuously.
   // When you bring a new item in front of it, the first frames are captured
   // with settings tuned for the previous scene. Discarding a few lets them
   // converge. This also flushes the stale frame sitting in the second buffer.
   for (int i = 0; i < WARMUP_FRAMES; i++) {
     camera_fb_t *fb = esp_camera_fb_get();
     if (fb) esp_camera_fb_return(fb);
   }
 
   Serial.printf("Capturing item %s burst %d...\n", currentItem.c_str(), burst);
   uint32_t start = millis();
   size_t totalBytes = 0;
   int saved = 0;
 
   for (int i = 0; i < FRAMES_PER_BURST; i++) {
     // Schedule against the burst start, not the previous frame, so slow SD
     // writes don't accumulate and stretch the burst.
     uint32_t target = start + (uint32_t)i * FRAME_INTERVAL_MS;
     while ((int32_t)(millis() - target) < 0) delay(1);
 
     camera_fb_t *fb = esp_camera_fb_get();
     if (!fb) {
       Serial.printf("  frame %d: capture failed\n", i);
       continue;
     }
 
     String path = framePath(burst, i);
     File f = SD.open(path, FILE_WRITE);
     if (!f) {
       Serial.printf("  frame %d: could not open %s\n", i, path.c_str());
       esp_camera_fb_return(fb);  // always give the buffer back, or the driver stalls
       continue;
     }
     size_t written = f.write(fb->buf, fb->len);
     f.close();
 
     if (written != fb->len) {
       Serial.printf("  frame %d: short write (%u of %u bytes)\n", i,
                     (unsigned)written, (unsigned)fb->len);
     } else {
       totalBytes += fb->len;
       saved++;
     }
     esp_camera_fb_return(fb);
   }
 
   uint32_t elapsed = millis() - start;
   lastBurst = burst;
   Serial.printf("OK item %s burst %d: %d/%d frames, %u KB, %lu ms\n",
                 currentItem.c_str(), burst, saved, FRAMES_PER_BURST,
                 (unsigned)(totalBytes / 1024), (unsigned long)elapsed);
   Serial.printf("   log row: %s,%d,<form>,<orientation>,<lighting>,<background>,%d\n",
                 currentItem.c_str(), burst, saved);
 }
 
 void undoBurst() {
   if (currentItem == "" || lastBurst < 0) {
     Serial.println("ERR: nothing to undo for this item");
     return;
   }
   int removed = 0;
   for (int i = 0; i < FRAMES_PER_BURST; i++) {
     String path = framePath(lastBurst, i);
     if (SD.exists(path) && SD.remove(path)) removed++;
   }
   Serial.printf("Deleted burst %d of item %s (%d files)\n",
                 lastBurst, currentItem.c_str(), removed);
   lastBurst = -1;  // only one level of undo, so you can't delete further by accident
 }
 
 // ---------------------------------------------------------------------------
 // Commands
 // ---------------------------------------------------------------------------
 void setItem(String id) {
   id.trim();
   // Require exactly 4 digits so filenames sort correctly and match the manifest.
   bool valid = id.length() == 4;
   for (unsigned i = 0; valid && i < id.length(); i++) {
     if (!isDigit(id[i])) valid = false;
   }
   if (!valid) {
     Serial.println("ERR: item ID must be 4 digits, e.g. `item 0001`");
     return;
   }
   currentItem = id;
   lastBurst = -1;
   if (!SD.exists(itemDir()) && !SD.mkdir(itemDir())) {
     Serial.printf("ERR: could not create %s\n", itemDir().c_str());
     currentItem = "";
     return;
   }
   Serial.printf("Item %s ready, next burst: %d\n", currentItem.c_str(), nextBurstNumber());
 }
 
 void listItem() {
   if (currentItem == "") {
     Serial.println("ERR: no item set");
     return;
   }
   File dir = SD.open(itemDir());
   if (!dir) {
     Serial.println("ERR: could not open item directory");
     return;
   }
   int count = 0;
   File f = dir.openNextFile();
   while (f) {
     Serial.printf("  %s  %u bytes\n", f.name(), (unsigned)f.size());
     count++;
     f = dir.openNextFile();
   }
   Serial.printf("%d files\n", count);
 }
 
 void printStatus() {
   Serial.printf("Item: %s\n", currentItem == "" ? "(none)" : currentItem.c_str());
   if (currentItem != "") Serial.printf("Next burst: %d\n", nextBurstNumber());
   Serial.printf("SD free: %llu MB\n",
                 (SD.totalBytes() - SD.usedBytes()) / (1024ULL * 1024ULL));
   Serial.printf("PSRAM free: %u KB\n", (unsigned)(ESP.getFreePsram() / 1024));
 }
 
 void printHelp() {
   Serial.println("Commands: item <4 digits> | cap | undo | ls | status | help");
 }
 
 void handleCommand(String line) {
   line.trim();
   if (line.length() == 0) return;
 
   if (line.startsWith("item ")) setItem(line.substring(5));
   else if (line == "cap") captureBurst();
   else if (line == "undo") undoBurst();
   else if (line == "ls") listItem();
   else if (line == "status") printStatus();
   else if (line == "help") printHelp();
   else Serial.printf("Unknown command: %s (try `help`)\n", line.c_str());
 }
 
 // ---------------------------------------------------------------------------
 // Arduino entry points
 // ---------------------------------------------------------------------------
 void setup() {
   Serial.begin(115200);
   delay(2000);  // native USB: give the Serial Monitor time to reconnect after reset
   Serial.println("\n=== BinSight capture ===");
 
   if (!psramFound()) {
     Serial.println("FATAL: no PSRAM. Set Tools -> PSRAM -> OPI PSRAM and re-upload.");
     return;  // loop() will still run, but commands will fail safely
   }
   Serial.printf("PSRAM: %u KB\n", (unsigned)(ESP.getPsramSize() / 1024));
 
   if (!initCamera()) {
     Serial.println("FATAL: camera init failed. Check the camera board is seated.");
     return;
   }
   Serial.println("Camera: OK (QVGA JPEG)");
 
   if (!SD.begin(SD_CS_PIN)) {
     Serial.println("FATAL: SD mount failed. Check the card is inserted and FAT32.");
     return;
   }
   Serial.printf("SD: OK, %llu MB free\n",
                 (SD.totalBytes() - SD.usedBytes()) / (1024ULL * 1024ULL));
 
   printHelp();
 }
 
 void loop() {
   // Non-blocking line reader: collect characters until a newline, then act.
   // Reading this way (rather than Serial.readStringUntil) never stalls loop(),
   // which matters later when the board also has to listen to the Uno.
   while (Serial.available()) {
     char c = Serial.read();
     if (c == '\n' || c == '\r') {
       if (lineBuf.length() > 0) {
         handleCommand(lineBuf);
         lineBuf = "";
       }
     } else if (lineBuf.length() < 64) {
       lineBuf += c;
     }
   }
 }