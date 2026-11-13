// ESP32 + HC-SR04 smart-bin node for Wokwi (https://wokwi.com).  UNTESTED here: verify in Wokwi.
// Wiring: HC-SR04 VCC->VIN(5V) GND->GND TRIG->D5 ECHO->D18.   Libraries: PubSubClient
#include <WiFi.h>
#include <PubSubClient.h>

const char* SSID = "Wokwi-GUEST";            // Wokwi's simulated WiFi
const char* MQTT_HOST = "broker.hivemq.com"; // public broker for the demo (or your own via a tunnel)
const char* BIN_ID = "BIN-001";
const int TRIG = 5, ECHO = 18;
const float BIN_HEIGHT_CM = 100.0;           // empty-bin sensor-to-floor distance

WiFiClient net; PubSubClient mqtt(net);
unsigned long seqNo = 0;

float readDistanceCm() {
  digitalWrite(TRIG, LOW); delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); delayMicroseconds(10); digitalWrite(TRIG, LOW);
  long us = pulseIn(ECHO, HIGH, 30000);
  return us * 0.0343 / 2.0;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  WiFi.begin(SSID, "", 6);
  while (WiFi.status() != WL_CONNECTED) { delay(250); Serial.print("."); }
  mqtt.setServer(MQTT_HOST, 1883);
}

void loop() {
  if (!mqtt.connected()) { mqtt.connect("wokwi-bin-001"); }
  float d = readDistanceCm();
  float fill = constrain(1.0 - d / BIN_HEIGHT_CM, 0.0, 1.0);
  char topic[64], msg[160];
  snprintf(topic, sizeof(topic), "smartwaste/bin/%s/telemetry", BIN_ID);
  snprintf(msg, sizeof(msg), "{\"bin_id\":\"%s\",\"seq\":%lu,\"fill\":%.3f,\"dist_cm\":%.1f}", BIN_ID, ++seqNo, fill, d);
  mqtt.publish(topic, msg);
  Serial.println(msg);
  mqtt.loop();
  delay(5000);   // use deep-sleep + longer period on real hardware (energy model)
}
