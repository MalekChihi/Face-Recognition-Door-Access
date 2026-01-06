#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Ascon128.h>       // 📦 Install "Ascon" by Rhys Weatherley via Library Manager
#include <mbedtls/base64.h> // Built-in ESP32 library

// --- WIFI & MQTT CONFIG ---
const char* WIFI_SSID = "Galaxy A14 158B";
const char* WIFI_PASS = "12345678";

const char* MQTT_BROKER = "10.171.181.60"; 
const int MQTT_PORT = 8883;
const char* MQTT_USER = "espuser";
const char* MQTT_PASS = "esppass";
const char* MQTT_TOPIC = "access_control/decision";

// --- 🔐 SECURITY KEY (MUST MATCH PYTHON EXACTLY) ---
// Key: 000102030405060708090A0B0C0D0E0F
const uint8_t ASCON_KEY[16] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F};

// CA certificate
const char* ca_cert = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDETCCAfmgAwIBAgIUHDAKemcBisJhWV0x6dI++gUFVTIwDQYJKoZIhvcNAQEL
BQAwGDEWMBQGA1UEAwwNMTAuMTcxLjE4MS42MDAeFw0yNjAxMDYwODE3NTVaFw0y
NzAxMDYwODE3NTVaMBgxFjAUBgNVBAMMDTEwLjE3MS4xODEuNjAwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQDdY8U523QP1d33xTETxrNbaku7x0sPno0o
iMm4p0pGD3be3OIFYGDjO1pO2C5S24rKfzbD4qWpBWXsfPKa1XGZhrCkxZRqJ1LT
n1G6qwZ/AcGx8SdBqY3WRKvX2ih6VkBcSuOd39CJvP2wjbCUcRNl5gy9R22wseYb
iuWHObmHw4XXNr0KMOsxWLHDEym5/3C1zJR53N3tgjP332l3sP1L+b+3ZEdNfJyR
of/BLWratQat/44g6ZUZBEOK8a4ouZb/DlEbXBx+BVDrU+l0oSBaV29oxBvWQlZQ
lqjtfxcP53ru/Vi2ywDZ2PBTcP4HRdzrCc2bKlVUifN66AVzwrNdAgMBAAGjUzBR
MB0GA1UdDgQWBBSEmxHmzW/B8h7DmLaKfP7EqhK1oDAfBgNVHSMEGDAWgBSEmxHm
zW/B8h7DmLaKfP7EqhK1oDAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUA
A4IBAQBNptHV156+DSxjnvBH0OrTfolhXwzoiY7wjU6CGh/HT06s/hbUJ695IIbr
Aq1BAiATKq1RX4ITaON1x2p8RdJ73rfOaIk23hZQ9E/9z+wxSwOgzfV2uEombxbe
4S24zIYMDUfFDKiBeYlmblEpMf/5Z/eGWwdapiJWWEzOnw+uMeQ9GRcQ2xTgx88K
F/iWw12cxDd/avHQ+2OrIb4Ug+TEkAx2egIw8Y2hAaHn9EYKQiknrgjgZC4yO0ZH
inqwLdCgjKN1LH89p51ST9cazUEYNMOggtenvy668fVBriESK8L2n18a86Rbksjd
mdoIPiHlmq4Ub1XN7yE7t7JO4J+W
-----END CERTIFICATE-----

)EOF";

// --- PINS (Using the corrected wiring from previous steps) ---
const byte PIN_RGB_R  = 26; 
const byte PIN_RGB_G  = 27; 
const byte PIN_RGB_B  = 14;

// --- STATE MANAGEMENT ---
int systemState = 0; // 0=Blue, 1=Green, 2=Red

WiFiClientSecure wifiClient;
PubSubClient client(wifiClient);

// --- RGB HELPER ---
void setRGB(bool r, bool g, bool b) {
  digitalWrite(PIN_RGB_R, r);
  digitalWrite(PIN_RGB_G, g);
  digitalWrite(PIN_RGB_B, b);
}

#define BLUE  setRGB(0, 0, 1)
#define GREEN setRGB(0, 1, 0)
#define RED   setRGB(1, 0, 0)

// --- 🔐 ASCON DECRYPTION HELPER ---
String decryptASCON(String encryptedBase64) {
  // 1. Decode Base64 to binary
  size_t outputLength;
  unsigned char decodedData[256]; // Buffer for raw binary
  
  // Use ESP32 built-in Base64 decoder
  int ret = mbedtls_base64_decode(decodedData, 256, &outputLength, 
                        (const unsigned char*)encryptedBase64.c_str(), encryptedBase64.length());

  if (ret != 0 || outputLength < 16) {
    Serial.println("❌ Base64 Decode Failed or too short");
    return ""; 
  }

  // 2. Extract Nonce (First 16 bytes)
  uint8_t nonce[16];
  memcpy(nonce, decodedData, 16);

  // 3. Extract Ciphertext (The rest)
  uint8_t ciphertext[200];
  size_t cipherLen = outputLength - 16;
  memcpy(ciphertext, decodedData + 16, cipherLen);

  // 4. Decrypt using Ascon128
  Ascon128 cipher;
  cipher.setKey(ASCON_KEY, 16);
  cipher.setIV(nonce, 16);
  
  // Ascon decrypts into a buffer
  uint8_t plaintext[200];
  cipher.decrypt(plaintext, ciphertext, cipherLen);

  // 5. Create String (ASCON tag is 16 bytes at end, we remove it to get text)
  // Ensure we don't read past buffer
  if (cipherLen < 16) return "";
  plaintext[cipherLen - 16] = '\0'; 
  
  return String((char*)plaintext);
}

// --- MQTT CALLBACK ---
void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String encryptedMessage;
    for (unsigned int i = 0; i < length; i++) encryptedMessage += (char)payload[i];

    // Debug: Show encrypted data (Proof of security)
    Serial.print("🔒 Cipher: ");
    Serial.println(encryptedMessage.substring(0, 20) + "..."); 

    // 1. DECRYPT
    String jsonString = decryptASCON(encryptedMessage);
    
    if (jsonString == "") {
      // Serial.println("❌ Decryption Error"); // Optional
      return;
    }
    
    // Debug: Show decrypted JSON
    // Serial.println("🔓 Plain: " + jsonString); 

    // 2. PARSE JSON
    StaticJsonDocument<256> doc;
    if (deserializeJson(doc, jsonString) != DeserializationError::Ok) return;

    const char* decision = doc["decision"];

    // 3. EXECUTE LOGIC (Event-Driven)
    
    // ACCESS GRANTED -> Green
    if (strcmp(decision, "GRANTED") == 0) {
        if (systemState != 1) {
            Serial.println("STATUS: ACCESS GRANTED (Green)");
            GREEN;
            systemState = 1;
        }
    }
    // ACCESS DENIED -> Red
    else if (strcmp(decision, "DENIED") == 0) {
        if (systemState != 2) {
            Serial.println("STATUS: ACCESS DENIED (Red)");
            RED;
            systemState = 2;
        }
    }
    // NO ONE -> Blue
    else if (strcmp(decision, "NO_ONE") == 0) {
        if (systemState != 0) {
            Serial.println("STATUS: IDLE (Blue)");
            BLUE;
            systemState = 0;
        }
    }
}

// --- CONNECT HELPERS ---
void reconnectMQTT() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32_SmartLock")) {
      Serial.println(" Connected!");
      client.subscribe(MQTT_TOPIC);
    } else {
      Serial.println(" Failed, retrying...");
      delay(2000);
    }
  }
}

void setupWiFi() {
  Serial.println("Connecting WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("WiFi Connected!");
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_RGB_R, OUTPUT);
  pinMode(PIN_RGB_G, OUTPUT);
  pinMode(PIN_RGB_B, OUTPUT);

  // Initial State: Blue
  BLUE; 

  setupWiFi();
  wifiClient.setCACert(ca_cert);

  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(mqttCallback);
}

void loop() {
  if (!client.connected()) reconnectMQTT();
  client.loop();
}