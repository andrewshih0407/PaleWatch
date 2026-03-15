#include <DHT.h>
#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

#define DHT_PIN       2
#define DHT_TYPE      DHT11
#define WATER_PIN     A0
#define TRIG_PIN      7
#define ECHO_PIN      6
#define LED_GREEN     52
#define LED_YELLOW    50
#define LED_RED       48
#define SERVO_PIN     25
#define BUZZER_PIN    46

#define TEMP_CRITICAL   29.0
#define HUMID_CRITICAL  85.0
#define WATER_CRITICAL  500
#define DIST_CRITICAL   5.0

DHT               dht(DHT_PIN, DHT_TYPE);
Servo             vent;
LiquidCrystal_I2C lcd(0x27, 16, 2);

unsigned long lastRead   = 0;
unsigned long lastScroll = 0;
#define READ_INTERVAL    2000
#define SCROLL_INTERVAL  3000

int   lcdPage      = 0;
float lastTemp     = 0;
float lastHumidity = 0;
int   lastWater    = 0;
float lastDist     = 0;
int   lastScore    = 0;

void setup() {
  Serial.begin(9600);
  delay(2000);
  Serial.println("PaleWatch booting...");

  dht.begin();
  Serial.println("DHT started");

  vent.attach(SERVO_PIN);
  vent.write(0);

  pinMode(TRIG_PIN,   OUTPUT);
  pinMode(ECHO_PIN,   INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_GREEN,  OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED,    OUTPUT);

  noTone(BUZZER_PIN);
  digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    LOW);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("PaleWatch  v4.0");
  lcd.setCursor(0, 1); lcd.print("Starting up...");

  digitalWrite(LED_GREEN,  HIGH); delay(300); digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_YELLOW, HIGH); delay(300); digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    HIGH); delay(300); digitalWrite(LED_RED,    LOW);
  tone(BUZZER_PIN, 1000); delay(200); noTone(BUZZER_PIN);

  delay(500);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("PaleWatch Ready");
  lcd.setCursor(0, 1); lcd.print("Reading sensors");
  delay(1000);
  lcd.clear();

  Serial.println("Setup complete");
}

void loop() {
  unsigned long now = millis();

  if (now - lastRead >= READ_INTERVAL) {
    lastRead = now;

    float temp     = dht.readTemperature();
    float humidity = dht.readHumidity();
    int   water    = analogRead(WATER_PIN);
    float distance = readUltrasonic();

    if (isnan(temp))     temp     = 0.0;
    if (isnan(humidity)) humidity = 0.0;

    lastTemp     = temp;
    lastHumidity = humidity;
    lastWater    = water;
    lastDist     = distance;

    int score = 0;
    if (temp     >= TEMP_CRITICAL)  score++;
    if (humidity >= HUMID_CRITICAL) score++;
    if (water    >= WATER_CRITICAL) score++;
    if (distance <= DIST_CRITICAL)  score++;
    lastScore = score;

    driveOutputs(score);

    Serial.print("{\"temp\":");
    Serial.print(temp, 1);
    Serial.print(",\"humidity\":");
    Serial.print(humidity, 1);
    Serial.print(",\"water\":");
    Serial.print(water);
    Serial.print(",\"distance\":");
    Serial.print(distance, 1);
    Serial.print(",\"score\":");
    Serial.print(score);
    Serial.println("}");
  }

  if (now - lastScroll >= SCROLL_INTERVAL) {
    lastScroll = now;
    lcdPage = (lcdPage + 1) % 3;
    updateLCD();
  }
}

void driveOutputs(int score) {
  digitalWrite(LED_GREEN,  LOW);
  digitalWrite(LED_YELLOW, LOW);
  digitalWrite(LED_RED,    LOW);

  if      (score >= 3) digitalWrite(LED_RED,    HIGH);
  else if (score >= 1) digitalWrite(LED_YELLOW, HIGH);
  else                 digitalWrite(LED_GREEN,  HIGH);

  if (score >= 3) tone(BUZZER_PIN, 1000);
  else            noTone(BUZZER_PIN);

  if      (score >= 3) vent.write(90);
  else if (score >= 1) vent.write(45);
  else                 vent.write(0);

  updateLCD();
}

void updateLCD() {
  lcd.clear();

  if (lcdPage == 0) {
    lcd.setCursor(0, 0);
    lcd.print("Reef Status:");
    lcd.setCursor(0, 1);
    if      (lastScore >= 3) lcd.print("!! CRITICAL  !!");
    else if (lastScore >= 1) lcd.print("   WARNING      ");
    else                     lcd.print("   HEALTHY      ");

  } else if (lcdPage == 1) {
    lcd.setCursor(0, 0);
    lcd.print("Temp: ");
    lcd.print(lastTemp, 1);
    lcd.print((char)223);
    lcd.print("C");
    lcd.setCursor(0, 1);
    lcd.print("Humid:");
    lcd.print(lastHumidity, 1);
    lcd.print("%");

  } else {
    lcd.setCursor(0, 0);
    lcd.print("Water:");
    lcd.print(lastWater);
    lcd.setCursor(0, 1);
    lcd.print("Dist: ");
    lcd.print(lastDist, 1);
    lcd.print("cm");
  }
}

float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  long dur = pulseIn(ECHO_PIN, HIGH, 30000);
  if (dur == 0) return 999.0;
  return dur * 0.034 / 2.0;
}
