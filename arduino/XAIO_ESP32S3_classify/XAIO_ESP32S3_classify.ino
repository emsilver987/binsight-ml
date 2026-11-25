/*
 * BinSight classify firmware — XIAO ESP32-S3 Sense
 *
 * A copy of XAIO_ESP32S3.ino that also runs the int8 MobileNet on the camera.
 * `cap` still saves a JPEG burst to the microSD card. `sort` grabs a burst,
 * averages the frame probabilities, applies the rules.yaml thresholds, and
 * prints the material and stream on Serial.
 *
 * Commands (115200 baud, line ending "Newline"):
 *   item 0001   set the current item ID (creates /item0001/ on the card)
 *   cap         capture one burst of FRAMES_PER_BURST frames for the current item
 *   sort        classify the item in front of the camera and print the decision
 *   auto        keep classifying and send each decision to the Uno
 *   stop        pause that automatic classify loop
 *   undo        delete the last burst you captured
 *   ls          list the files for the current item
 *   status      show current item, next burst number, free space
 *   help        show this list
 *
 * Uno link: D6 (TX) to Uno pin 0 (RX), grounds tied. 115200.
 * Each decision is also one line on D6: SORT <code> <material> <conf> <display> <frames> <ms>
 * Re-upload this sketch before watching the Uno. The previous firmware never transmitted.
 *
 * Open this folder in the Arduino IDE, not arduino/XAIO_ESP32S3.ino.
 * The IDE compiles every .ino in a folder as one sketch.
 *
 * Board: XIAO_ESP32S3. Tools -> PSRAM -> OPI PSRAM.
 * TensorFlow Lite Micro comes with the ESP32 board package. Do not install
 * a separate TFLite library; an older one conflicts with this core.
 *
 * Regenerate the model headers after retraining, from binsight-ml/:
 *   .venv/bin/python src/export_tflite.py --config configs/binsight.yaml --run runs/binsight
 */

#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <esp_heap_caps.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "binsight_model.h"
#include "binsight_labels.h"

// ---------------------------------------------------------------------------
// Capture settings — change these, not the code below
// ---------------------------------------------------------------------------
const int FRAMES_PER_BURST = 5;     // 3 bursts x 5 frames = 15 frames per item
const int FRAME_INTERVAL_MS = 200;  // 5 fps
const int WARMUP_FRAMES = 4;        // discarded before each burst (see captureBurst)
const int JPEG_QUALITY = 10;        // 0-63, lower = better quality, bigger files
const int SD_CS_PIN = 21;           // XIAO Sense microSD chip select
const size_t kArenaBytes = 1024 * 1024;  // PSRAM working memory for the int8 model
const int LINK_RX_PIN = 44;         // D7. Leave it unconnected; Uno TX is 5 V.
const int LINK_TX_PIN = 43;         // D6 -> Uno pin 0 (RX)
const uint32_t AUTO_GAP_MS = 3000;  // pause between automatic sorts

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
bool sdReady = false;
bool modelReady = false;
bool autoSort = true;          // headless: the Uno is the only serial monitor
uint32_t lastSortMs = 0;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *inputTensor = nullptr;
TfLiteTensor *outputTensor = nullptr;

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
  // browser could switch resolutions). Smaller buffers leave PSRAM for the model.
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
// Model
// ---------------------------------------------------------------------------
bool initModel() {
  const tflite::Model *model = tflite::GetModel(TFLM_binsight_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf(
        "FATAL: model schema %u, runtime supports %d. Re-export the model.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // The tensor arena is far too big for internal RAM. The camera buffers
  // already live in PSRAM, and so does this.
  uint8_t *arena = (uint8_t *)heap_caps_aligned_alloc(
      16, kArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!arena) {
    Serial.println("FATAL: could not allocate the tensor arena in PSRAM.");
    return false;
  }

  static auto resolver = TFLMgetResolver();
  static tflite::MicroInterpreter staticInterpreter(model, resolver, arena, kArenaBytes);
  if (staticInterpreter.AllocateTensors() != kTfLiteOk) {
    Serial.printf("FATAL: AllocateTensors failed. Arena is %u KB.\n",
                  (unsigned)(kArenaBytes / 1024));
    return false;
  }

  interpreter = &staticInterpreter;
  inputTensor = interpreter->input(0);
  outputTensor = interpreter->output(0);
  if (inputTensor->type != kTfLiteInt8 || outputTensor->type != kTfLiteInt8) {
    Serial.println("FATAL: model input and output must be int8.");
    return false;
  }
  size_t expectBytes = (size_t)kImageSize * kImageSize * 3;
  if (inputTensor->bytes != expectBytes || outputTensor->bytes != kClassCount) {
    Serial.printf("FATAL: tensor sizes input %u output %u.\n",
                  (unsigned)inputTensor->bytes, (unsigned)outputTensor->bytes);
    return false;
  }
  if (inputTensor->params.scale == 0.0f || outputTensor->params.scale == 0.0f) {
    Serial.println("FATAL: missing quantization scale.");
    return false;
  }

  Serial.printf("Model: %u KB int8, arena used %u KB\n",
                (unsigned)(TFLMmodelLength / 1024),
                (unsigned)(interpreter->arena_used_bytes() / 1024));
  modelReady = true;
  return true;
}

// Center-crop to a square, bilinear-resize to 96x96, then the same
// MobileNet scaling the laptop uses: pixel/127.5 - 1, quantized to int8.
bool prepareInput(const uint8_t *rgb, int width, int height) {
  int side = width < height ? width : height;
  int originX = (width - side) / 2;
  int originY = (height - side) / 2;
  int8_t *dest = inputTensor->data.int8;
  float scale = inputTensor->params.scale;
  int zeroPoint = inputTensor->params.zero_point;

  for (int y = 0; y < kImageSize; y++) {
    float srcY = ((y + 0.5f) * side) / kImageSize - 0.5f;
    int y0 = (int)floorf(srcY);
    float yWeight = srcY - y0;
    if (y0 < 0) {
      y0 = 0;
      yWeight = 0.0f;
    }
    int y1 = y0 + 1;
    if (y1 >= side) y1 = side - 1;
    if (y0 >= side) y0 = side - 1;

    for (int x = 0; x < kImageSize; x++) {
      float srcX = ((x + 0.5f) * side) / kImageSize - 0.5f;
      int x0 = (int)floorf(srcX);
      float xWeight = srcX - x0;
      if (x0 < 0) {
        x0 = 0;
        xWeight = 0.0f;
      }
      int x1 = x0 + 1;
      if (x1 >= side) x1 = side - 1;
      if (x0 >= side) x0 = side - 1;

      for (int channel = 0; channel < 3; channel++) {
        float p00 = rgb[((originY + y0) * width + (originX + x0)) * 3 + channel];
        float p01 = rgb[((originY + y0) * width + (originX + x1)) * 3 + channel];
        float p10 = rgb[((originY + y1) * width + (originX + x0)) * 3 + channel];
        float p11 = rgb[((originY + y1) * width + (originX + x1)) * 3 + channel];
        float top = p00 + (p01 - p00) * xWeight;
        float bottom = p10 + (p11 - p10) * xWeight;
        float pixel = top + (bottom - top) * yWeight;
        int quantized = (int)lroundf((pixel / 127.5f - 1.0f) / scale + zeroPoint);
        if (quantized < -128) quantized = -128;
        if (quantized > 127) quantized = 127;
        *dest++ = (int8_t)quantized;
      }
    }
  }
  return true;
}

void sortItem() {
  if (!modelReady) {
    Serial.println("ERR: model is not ready");
    return;
  }

  for (int i = 0; i < WARMUP_FRAMES; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }

  float classScores[kClassCount];
  memset(classScores, 0, sizeof(classScores));
  int used = 0;
  uint32_t inferMs = 0;
  uint32_t start = millis();
  Serial.println("Sorting...");

  for (int frame = 0; frame < kVoteWindow; frame++) {
    uint32_t target = start + (uint32_t)frame * FRAME_INTERVAL_MS;
    while ((int32_t)(millis() - target) < 0) delay(1);

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.printf("  frame %d: capture failed\n", frame);
      continue;
    }
    size_t rgbBytes = (size_t)fb->width * fb->height * 3;
    uint8_t *rgb = (uint8_t *)heap_caps_malloc(rgbBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb) {
      Serial.println("ERR: no PSRAM for the RGB buffer");
      esp_camera_fb_return(fb);
      return;
    }
    bool decoded = fmt2rgb888(fb->buf, fb->len, fb->format, rgb);
    int width = fb->width;
    int height = fb->height;
    esp_camera_fb_return(fb);
    if (!decoded) {
      Serial.printf("  frame %d: JPEG decode failed\n", frame);
      heap_caps_free(rgb);
      continue;
    }
    prepareInput(rgb, width, height);
    heap_caps_free(rgb);

    uint32_t began = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.printf("  frame %d: invoke failed\n", frame);
      continue;
    }
    inferMs += millis() - began;

    float scale = outputTensor->params.scale;
    int zeroPoint = outputTensor->params.zero_point;
    for (int klass = 0; klass < kClassCount; klass++) {
      classScores[klass] += (outputTensor->data.int8[klass] - zeroPoint) * scale;
    }
    used++;
  }

  if (used == 0) {
    Serial.println("ERR: no frames classified");
    return;
  }
  for (int klass = 0; klass < kClassCount; klass++) classScores[klass] /= used;

  int material = 0;
  for (int klass = 1; klass < kClassCount; klass++) {
    if (classScores[klass] > classScores[material]) material = klass;
  }

  float streamScores[kStreamCount];
  memset(streamScores, 0, sizeof(streamScores));
  for (int klass = 0; klass < kClassCount; klass++) {
    streamScores[kClassToStream[klass]] += classScores[klass];
  }
  int stream = 0;
  for (int index = 1; index < kStreamCount; index++) {
    if (streamScores[index] > streamScores[stream]) stream = index;
  }

  float confidence = streamScores[stream];
  bool accept = confidence + 0.000001f >= kStreamThresholds[stream];
  char code = accept ? kStreamCodes[stream] : kUnknownCode;
  const char *streamName = accept ? kStreamNames[stream] : "Unknown";
  const char *display = accept ? kStreamDisplays[stream] : kUnknownDisplay;
  unsigned long msPerFrame = (unsigned long)(inferMs / used);

  Serial.println("--- sort ---");
  Serial.printf("material: %s\n", kClassNames[material]);
  Serial.printf("stream: %s\n", streamName);
  if (accept) {
    Serial.printf("confidence: %.3f\n", confidence);
  } else {
    Serial.printf("confidence: %.3f  below %s %.2f\n",
                  confidence, kStreamNames[stream], kStreamThresholds[stream]);
  }
  Serial.printf("code: %c\n", code);
  Serial.printf("display: %s\n", display);
  Serial.printf("frames: %d\n", used);
  Serial.printf("ms/frame: %lu\n", msPerFrame);

  // Same decision, one line, out D6 to the Uno.
  Serial1.printf("SORT %c %s %.3f %s %d %lu\n",
                 code, kClassNames[material], confidence, display, used, msPerFrame);
  lastSortMs = millis();
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
  if (!sdReady) {
    Serial.println("ERR: SD card is not mounted");
    return;
  }
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
  if (!sdReady) {
    Serial.println("ERR: SD card is not mounted");
    return;
  }
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
  if (!sdReady) {
    Serial.println("ERR: SD card is not mounted");
    return;
  }
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
  if (!sdReady) {
    Serial.println("ERR: SD card is not mounted");
    return;
  }
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
  if (sdReady && currentItem != "") Serial.printf("Next burst: %d\n", nextBurstNumber());
  if (sdReady) {
    Serial.printf("SD free: %llu MB\n",
                  (SD.totalBytes() - SD.usedBytes()) / (1024ULL * 1024ULL));
  } else {
    Serial.println("SD: not mounted");
  }
  Serial.printf("PSRAM free: %u KB\n", (unsigned)(ESP.getFreePsram() / 1024));
  Serial.printf("Model: %s\n", modelReady ? "ready" : "not ready");
}

void printHelp() {
  Serial.println("Commands: item <4 digits> | cap | sort | auto | stop | undo | ls | status | help");
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("item ")) setItem(line.substring(5));
  else if (line == "cap") captureBurst();
  else if (line == "sort") sortItem();
  else if (line == "auto") {
    autoSort = true;
    Serial.println("Auto-sort on. Decisions go out D6 to the Uno.");
  } else if (line == "stop") {
    autoSort = false;
    Serial.println("Auto-sort paused. Type `sort` or `auto`.");
  } else if (line == "undo") undoBurst();
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
  Serial1.begin(115200, SERIAL_8N1, LINK_RX_PIN, LINK_TX_PIN);
  delay(2000);  // native USB: give the Serial Monitor time to reconnect after reset
  Serial.println("\n=== BinSight classify ===");

  if (!psramFound()) {
    Serial.println("FATAL: no PSRAM. Set Tools -> PSRAM -> OPI PSRAM and re-upload.");
    Serial1.println("ERR psram");
    return;
  }
  Serial.printf("PSRAM: %u KB\n", (unsigned)(ESP.getPsramSize() / 1024));

  if (!initModel()) {
    Serial.println("FATAL: model init failed.");
    Serial1.println("ERR model");
    return;
  }

  if (!initCamera()) {
    Serial.println("FATAL: camera init failed. Check the camera board is seated.");
    Serial1.println("ERR camera");
    return;
  }
  Serial.println("Camera: OK (QVGA JPEG)");

  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD mount failed. `sort` still works; `cap` needs the card.");
  } else {
    sdReady = true;
    Serial.printf("SD: OK, %llu MB free\n",
                  (SD.totalBytes() - SD.usedBytes()) / (1024ULL * 1024ULL));
  }

  printHelp();
  Serial.println("Uno link on D6. Auto-sort is on (`stop` pauses it).");
  Serial1.println("READY");
  lastSortMs = millis();
}

void loop() {
  // Non-blocking line reader: collect characters until a newline, then act.
  // Reading this way (rather than Serial.readStringUntil) never stalls loop().
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

  // With the USB cable on the Uno, nobody can type `sort`. Classify on a timer
  // and let the Uno print it. `stop` pauses this while you use the XIAO monitor.
  if (autoSort && modelReady && lineBuf.length() == 0 &&
      (uint32_t)(millis() - lastSortMs) >= AUTO_GAP_MS) {
    sortItem();
  }
}
