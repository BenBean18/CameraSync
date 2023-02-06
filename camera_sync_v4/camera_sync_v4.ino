// A program to cycle through a <LED_COUNT> LED NeoPixel strip
// at <DELAY_MS>ms intervals. This can run on any Arduino-compatible
// board, although I'm using an ESP-12E.

// Created by Ben Goldberg in 2022

// include NeoPixel library
#include <Adafruit_NeoPixel.h>

// define constants:
#define LED_PIN    D8 // the pin the LED strip is connected to
#define LED_COUNT  60 // how many LEDs are attached

#define LED_PIN2    D7 // the pin the LED strip is connected to
#define LED_COUNT2  60 // how many LEDs are attached

// and changeable not constants:
int DELAY_MS = 1;     // how many milliseconds each LED should stay on
int LED_GAP = 0;      // how many off LEDs between each on LED (for spacing)

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800); // create the LED strip object
Adafruit_NeoPixel strip2(LED_COUNT2, LED_PIN2, NEO_GRBW + NEO_KHZ800); // create the LED strip object
void setup() {
  Serial.begin(115200);
  strip.begin(); // Initialize LED strip
  strip.show();  // Turn off all LEDs
  strip.setBrightness(255);
  strip.setPixelColor(0, strip.Color(0, 255, 0));
  strip.setPixelColor(LED_COUNT-1, strip.Color(0, 255, 0));
  
  strip2.begin(); // Initialize LED strip
  strip2.show();  // Turn off all LEDs
  strip2.setBrightness(255);
  strip2.setPixelColor(0, strip.Color(0, 255, 0));
  strip2.setPixelColor(LED_COUNT-1, strip.Color(0, 255, 0));
}

uint8_t color = 0;
const uint8_t MAX_COLOR_INDEX = 0; // length of colors - 1

// red is the lowest energy color so it will interfere less with the other LED strip
unsigned int colors[] = {strip.Color(255, 0, 0)/*, strip.Color(0, 255, 0), strip.Color(0, 0, 255), strip.Color(255, 255, 255)*/};
// red, green, blue, white

uint8_t bottom_pixel = 1;

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

  if (bottom_pixel == 1) {
    for (uint8_t p = LED_COUNT - 2; p >= (LED_COUNT-1) - LED_GAP - 1; p--) {
      strip2.setPixelColor(p, strip2.Color(0, 0, 0));
    }
  } else {
    strip2.setPixelColor(bottom_pixel - LED_GAP - 1, strip2.Color(0, 0, 0));
  }
  // turn it on and make it the current color
  strip2.setPixelColor(bottom_pixel, colors[color]);
  // update the LED strip
  strip2.show();
  bottom_pixel++;
  if (bottom_pixel == LED_COUNT-1) {
    bottom_pixel = 1;
  }

  color = color == MAX_COLOR_INDEX ? 0 : (color + 1);
  while (micros() - start < DELAY_MS*1000) {
    delayMicroseconds(1);
  }
}
