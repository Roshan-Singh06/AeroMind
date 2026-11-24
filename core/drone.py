
from dataclasses import dataclass

@dataclass
class Drone:
    mass: float = 3.0
    payload: float = 0.5
    max_thrust: float = 60.0
    battery_capacity: float = 100.0
    battery: float = 100.0
    drag_coefficient: float = 0.8
    reference_area: float = 0.12

    def total_mass(self):
        return self.mass + self.payload

    def consume_energy(self, power, dt):
        # Simplified battery energy model.
        energy_used = power * dt / 3600.0
        self.battery = max(
            0.0,
            self.battery - energy_used
        )

    def status(self):
        return {
            "Mass": self.total_mass(),
            "Battery": self.battery,
            "Maximum thrust": self.max_thrust
        }
