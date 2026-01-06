"""
MQTT Client for sending access control decisions to hardware.
"""

import paho.mqtt.client as mqtt
import json
import ascon    # pip install ascon
import base64
import os
from datetime import datetime
from typing import Optional

# --- 🔐 SECURITY CONFIGURATION ---
# This 16-byte key MUST match the one in your ESP32 code exactly.
# 000102030405060708090A0B0C0D0E0F
SECRET_KEY = bytes.fromhex("000102030405060708090A0B0C0D0E0F")

class MQTTPublisher:
    """
    MQTT Publisher for access control system.
    Publishes access decisions to hardware via MQTT broker.
    """

    def __init__(self, broker_address: str, port: int = 8883, 
                 topic: str = "access_control/decision",
                 client_id: str = "face_detection_system",
                 username: str = None,
                 password: str = None,
                 cafile: str = None):
        self.broker_address = broker_address
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.connected = False
        
        # MQTT client
        self.client = mqtt.Client(client_id=client_id)
        
        # Set authentication
        if username and password:
            self.client.username_pw_set(username, password)
        
        # Enable TLS
        if cafile:
            self.client.tls_set(ca_certs=cafile)
            self.client.tls_insecure_set(True)  # CRITICAL for self-signed certs
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        if rc == 0:
            self.connected = True
            print(f"📡 Connected to MQTT broker at {self.broker_address}:{self.port}")
        else:
            self.connected = False
            print(f"❌ MQTT connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker."""
        self.connected = False
        if rc != 0:
            print(f"⚠️  Unexpected MQTT disconnection (code {rc})")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        Returns: True if connection successful, False otherwise
        """
        try:
            self.client.connect(self.broker_address, self.port, keepalive=60)
            self.client.loop_start()  # Start background network loop
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            print("📡 Disconnected from MQTT broker")
    
    def publish_decision(self, user_name: Optional[str], decision: str, 
                        similarity: float = 0.0) -> bool:
        """
        Publish access control decision to MQTT broker.
        ENCRYPTED WITH ASCON-128
        """
        assert self.connected, "MQTT client is not connected"
        if not self.connected:
            print("⚠️  Cannot publish: Not connected to MQTT broker")
            return False
        
        # 1. Create message payload (Plaintext JSON)
        payload_dict = {
            "user": user_name if user_name else "UNKNOWN",
            "decision": decision,
            "similarity": round(similarity, 3),
            "timestamp": datetime.now().isoformat()
        }
        json_plaintext = json.dumps(payload_dict)
        
        try:
            # 2. 🔒 ENCRYPT WITH ASCON
            
            # A. Generate a unique 16-byte nonce (IV)
            nonce = os.urandom(16)
            
            # B. Encrypt the JSON string
            # Variant "Ascon-128" is the standard
            ciphertext = ascon.encrypt(
                SECRET_KEY, 
                nonce, 
                associateddata=b"", 
                plaintext=json_plaintext.encode('utf-8'), 
                variant="Ascon-128"
            )
            
            # C. Combine Nonce + Ciphertext and Encode in Base64
            # We send the nonce so the ESP32 knows how to decrypt it
            final_message = base64.b64encode(nonce + ciphertext).decode('utf-8')

            # 3. Publish the Encrypted String
            result = self.client.publish(
                self.topic,
                final_message,
                qos=1
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"🔒 Published (Encrypted): {decision} for {payload_dict['user']}")
                return True
            else:
                print(f"❌ Publish failed with code {result.rc}")
                return False
                
        except Exception as e:
            print(f"❌ Error during encryption/publishing: {e}")
            return False