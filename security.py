import hashlib
import os
import socket
import subprocess
import requests

class CyberSecurity:
    def __init__(self):
        self.firewall_status = False
        self.suspicious_ips = set(["192.168.1.100", "203.0.113.5"])  # Example blacklist

    def monitor_threats(self):
        print("[Security] Monitoring system for threats...")
        self.scan_network()
        self.check_firewall()
        self.detect_intrusions()

    def scan_network(self):
        """Scans the network for unauthorized access"""
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            print(f"[Security] Scanning local network from {local_ip}...")
            connected_devices = subprocess.check_output("arp -a", shell=True).decode()
            for ip in self.suspicious_ips:
                if ip in connected_devices:
                    print(f"[WARNING] Suspicious device detected: {ip}")
        except Exception as e:
            print(f"[Security] Network scan failed: {e}")

    def check_firewall(self):
        """Checks if the system firewall is active"""
        try:
            if os.name == "nt":  # Windows
                output = subprocess.check_output("netsh advfirewall show allprofiles", shell=True).decode()
                self.firewall_status = "ON" in output
            else:  # Linux/macOS
                output = subprocess.check_output("sudo ufw status", shell=True).decode()
                self.firewall_status = "active" in output.lower()
            
            print(f"[Security] Firewall Active: {self.firewall_status}")
        except Exception as e:
            print(f"[Security] Firewall check failed: {e}")

    def detect_intrusions(self):
        """Detects unauthorized login attempts or brute-force attacks"""
        try:
            log_path = "/var/log/auth.log" if os.name != "nt" else "C:\\Windows\\System32\\winevt\\Logs\\Security.evtx"
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
                    logs = log_file.readlines()[-50:]  # Check last 50 lines
                brute_force_attempts = [line for line in logs if "failed password" in line.lower()]
                if brute_force_attempts:
                    print(f"[WARNING] Intrusion attempt detected:\n{brute_force_attempts[-1]}")
            else:
                print("[Security] No intrusion logs found.")
        except Exception as e:
            print(f"[Security] Intrusion detection failed: {e}")

    def encrypt_data(self, data):
        """Encrypts sensitive data using SHA-256"""
        encrypted = hashlib.sha256(data.encode()).hexdigest()
        print(f"[Security] Encrypted Data: {encrypted}")
        return encrypted

    def check_vulnerabilities(self):
        """Checks for known security vulnerabilities"""
        try:
            response = requests.get("https://cve.circl.lu/api/last")  # Fetch latest security vulnerabilities
            if response.status_code == 200:
                latest_cves = response.json()[:5]
                for cve in latest_cves:
                    print(f"[CVE Alert] {cve['id']}: {cve['summary']}")
            else:
                print("[Security] Could not retrieve latest vulnerabilities.")
        except Exception as e:
            print(f"[Security] Vulnerability check failed: {e}")

# Example standalone execution
if __name__ == "__main__":
    security = CyberSecurity()
    security.monitor_threats()
    security.encrypt_data("SecureMessage123")
    security.check_vulnerabilities()
