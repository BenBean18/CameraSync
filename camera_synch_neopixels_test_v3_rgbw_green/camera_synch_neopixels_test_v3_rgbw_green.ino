// A program to cycle through a <LED_COUNT> LED NeoPixel strip
// at <DELAY_MS>ms intervals. This can run on any Arduino-compatible
// board, although I'm using an ESP-12E.

// Created by Ben Goldberg in 2022

// include NeoPixel library
#include <Adafruit_NeoPixel.h>

// include regex library
//#include <Regexp.h>

//// include WiFi libraries
//#include <ESP8266WiFi.h>
//#include <ESP8266mDNS.h>
//#include <WiFiClient.h>

// telnet to port 12345.
// structure: <command><value>
// <command> can be d (delayMs) or g (ledGap)
// e.g. d15 sets DELAY_MS = 15ms
//WiFiServer server(12345);
//WiFiClient client;

// define constants:
#define LED_PIN    D8 // the pin the LED strip is connected to
#define LED_COUNT  60 // how many LEDs are attached

// and changeable not constants:
int DELAY_MS = 25;     // how many milliseconds each LED should stay on
int LED_GAP = 0;      // how many off LEDs between each on LED (for spacing)

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRBW + NEO_KHZ800); // create the LED strip object
void setup() {
  Serial.begin(115200);
  strip.begin(); // Initialize LED strip
  strip.show();  // Turn off all LEDs
  strip.setBrightness(100); // Set to 100/255 = ~39% brightness
  strip.setPixelColor(0, strip.Color(0, 255, 0));
  strip.setPixelColor(LED_COUNT-1, strip.Color(0, 255, 0));
//  WiFi.mode(WIFI_STA);
//  WiFi.begin("SSID", "password");
//  Serial.print("Connecting to WiFi");
//  while (WiFi.status() != WL_CONNECTED) {
//    Serial.print(".");
//    delay(500);
//  }
//  Serial.print("Connected! IP address: ");
//  Serial.println(WiFi.localIP());
//
//  if (!MDNS.begin("ledStrip")) {
//    Serial.println("Error setting up MDNS responder!");
//    while (1) {
//      delay(1000);
//    }
//  }
//  Serial.println("mDNS responder started");
//  MDNS.addService("telnet", "tcp", 12345);
//  
//  server.begin();
}
//
//// telnet code modified from https://gist.github.com/tablatronix/4793677ca748f5f584c95ec4a2b10303
//void handleTelnet(){
//  if (server.hasClient()){
//    // client is connected
//    if (!client || !client.connected()){
//      if(client) client.stop(); // client disconnected
//      client = server.available(); // ready for new client
//    } else {
//      server.available().stop();  // have client, block new conections
//    }
//  }
//
//  if (client && client.connected() && client.available()){
//    handleMessage(client.readStringUntil('\n').c_str());
//  } 
//}
//
//void handleMessage(const char* data) {
//  MatchState ms((char*)data);
//  ms.Match("([dg])([0-9]+)");
//  char cap[30];
//  if (ms.level >= 2) {
//    ms.GetCapture(cap, 0);
//    switch (cap[0]) {
//      case 'd': // delayMs
//        ms.GetCapture(cap, 1);
//        DELAY_MS = atoi(cap);
//        break;
//      case 'g':
//        ms.GetCapture(cap, 1);
//        LED_GAP = atoi(cap);
//        break;
//    }
//  }
//}

uint8_t color = 0;
const uint8_t MAX_COLOR_INDEX = 0; // length of colors - 1

// red is the lowest energy color so it will interfere less with the other LED strip
unsigned int colors[] = {strip.Color(255, 0, 0)/*, strip.Color(0, 255, 0), strip.Color(0, 0, 255), strip.Color(0, 0, 0, 255)*/};
// red, green, blue, white

void loop() {
  unsigned long start;
  // For each pixel (separated by LED_GAP pixels),
  for (uint8_t pixel = 1; pixel < LED_COUNT-1; pixel += LED_GAP+1) {
    start = micros();
    // turn the pixel before it off
    if (pixel == 1) {
      for (uint8_t p = LED_COUNT - 2; p >= (LED_COUNT-1) - LED_GAP - 1; p--) {
        strip.setPixelColor(p, strip.Color(0, 0, 0));
      }
    } else {
      strip.setPixelColor(pixel - LED_GAP - 1, strip.Color(0, 0, 0));
    }
    // turn it on and make it the current color
    strip.setPixelColor(pixel, colors[color]);
    // update the LED strip
    strip.show();
    // wait DELAY_MS milliseconds
    if (pixel + LED_GAP + 1 < (LED_COUNT-1)) {
      while (micros() - start < DELAY_MS*1000) {
        delayMicroseconds(1);
      }
    }
  }
  color = color == MAX_COLOR_INDEX ? 0 : (color + 1);
//  handleTelnet();
//  MDNS.update();
  while (micros() - start < DELAY_MS*1000) {
    delayMicroseconds(1);
  }
}
