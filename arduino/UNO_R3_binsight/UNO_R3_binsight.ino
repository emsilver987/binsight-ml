/*
 * BinSight lid board — Arduino Uno R3
 *
 * The XIAO classifies the item and sends one line on D6:
 *   READY
 *   SORT R metal_can 0.892 RECYCLE 5 180
 *   SORT ? cardboard 0.520 ? 5 210
 *   ERR camera
 *
 * This sketch prints that decision on the Uno's Serial Monitor, the same
 * block the camera prints, and blinks the onboard LED so a live link is
 * visible even before you read the text. Servos are not driven yet.
 *
 * Wire (no solder):
 *   XIAO D6  -> Uno pin 0 (RX)
 *   XIAO GND -> Uno GND
 *   XIAO 5V  <- Uno 5V     (powers the camera from the Uno's USB)
 * Leave Uno pin 1 (TX) unconnected.
 *
 * Unplug the wire from pin 0 while uploading this sketch, then plug it back.
 * Board: Arduino Uno. Serial Monitor: 115200 baud.
 *
 * Open this folder in the Arduino IDE, not the XIAO sketch folder.
 */

const uint32_t WAIT_NOTE_MS = 2000;

char lineBuf[96];
uint8_t lineLen = 0;
bool cameraReady = false;
uint32_t lastWaitNote = 0;

void blinkLink() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(40);
  digitalWrite(LED_BUILTIN, LOW);
}

int lidForCode(char code) {
  if (code == 'R') return 0;
  if (code == 'T') return 1;
  if (code == 'C') return 2;
  return -1;  // S and ? keep every lid closed
}

const char *streamForCode(char code) {
  if (code == 'R') return "Recycle";
  if (code == 'T') return "Trash";
  if (code == 'C') return "Compost";
  if (code == 'S') return "Special";
  return "Unknown";
}

void printSort(char code, const char *material, float confidence,
               const char *display, int frames, unsigned long msPerFrame) {
  int lid = lidForCode(code);
  Serial.println("--- sort ---");
  Serial.print("material: ");
  Serial.println(material);
  Serial.print("stream: ");
  Serial.println(streamForCode(code));
  Serial.print("confidence: ");
  Serial.println(confidence, 3);
  Serial.print("code: ");
  Serial.println(code);
  Serial.print("display: ");
  Serial.println(display);
  Serial.print("frames: ");
  Serial.println(frames);
  Serial.print("ms/frame: ");
  Serial.println(msPerFrame);
  if (lid < 0) {
    Serial.println("lid: closed");
  } else {
    Serial.print("lid: ");
    Serial.print(lid);
    Serial.println(" open");
  }
  Serial.println();
}

void handleLine(char *line) {
  blinkLink();

  if (strcmp(line, "READY") == 0) {
    cameraReady = true;
    Serial.println("Camera connected.");
    Serial.println();
    return;
  }

  if (strncmp(line, "ERR", 3) == 0) {
    Serial.print("Camera error: ");
    Serial.println(line);
    return;
  }

  char code;
  char material[32];
  float confidence = 0;
  char display[16];
  int frames = 0;
  unsigned long msPerFrame = 0;
  int got = sscanf(line, "SORT %c %31s %f %15s %d %lu",
                   &code, material, &confidence, display, &frames, &msPerFrame);
  if (got == 6) {
    cameraReady = true;
    printSort(code, material, confidence, display, frames, msPerFrame);
    return;
  }

  Serial.print("Camera said: ");
  Serial.println(line);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== BinSight Uno ===");
  Serial.println("Waiting for the camera on pin 0...");
  lastWaitNote = millis();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }

  if (!cameraReady && (uint32_t)(millis() - lastWaitNote) >= WAIT_NOTE_MS) {
    Serial.println("still waiting — check D6 to pin 0, and the ground wire");
    lastWaitNote = millis();
  }
}
