from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

class DroneAI:
    def __init__(self, connection_string="127.0.0.1:14550"):
        """Initialize drone connection."""
        print("[DroneAI] Connecting to drone...")
        self.vehicle = connect(connection_string, wait_ready=True)

    def arm_and_takeoff(self, target_altitude=10):
        """Arms the drone and makes it take off to a specified altitude."""
        print("[DroneAI] Arming drone...")
        while not self.vehicle.is_armable:
            print("[DroneAI] Waiting for drone to initialize...")
            time.sleep(1)

        self.vehicle.mode = VehicleMode("GUIDED")
        self.vehicle.armed = True

        while not self.vehicle.armed:
            print("[DroneAI] Waiting for arming...")
            time.sleep(1)

        print(f"[DroneAI] Taking off to {target_altitude}m...")
        self.vehicle.simple_takeoff(target_altitude)

        while True:
            print(f"[DroneAI] Altitude: {self.vehicle.location.global_relative_frame.alt:.2f}m")
            if self.vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
                print("[DroneAI] Target altitude reached!")
                break
            time.sleep(1)

    def navigate_to(self, lat, lon, alt=10):
        """Navigate the drone to a specific GPS location."""
        print(f"[DroneAI] Navigating to ({lat}, {lon}, {alt}m)...")
        target_location = LocationGlobalRelative(lat, lon, alt)
        self.vehicle.simple_goto(target_location)

    def land(self):
        """Safely land the drone."""
        print("[DroneAI] Initiating landing sequence...")
        self.vehicle.mode = VehicleMode("LAND")
        while self.vehicle.armed:
            print("[DroneAI] Landing...")
            time.sleep(1)
        print("[DroneAI] Landed successfully.")

    def return_to_home(self):
        """Commands the drone to return to its home location."""
        print("[DroneAI] Returning to home location...")
        self.vehicle.mode = VehicleMode("RTL")

    def close_connection(self):
        """Closes the connection to the drone."""
        print("[DroneAI] Closing connection...")
        self.vehicle.close()
