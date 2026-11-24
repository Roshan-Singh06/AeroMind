
class PIDController:

    def __init__(self, kp=1.0, ki=0.01, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, target, current, dt):
        error = target - current

        self.integral += error * dt

        derivative = (
            (error - self.previous_error) / dt
            if dt > 0 else 0.0
        )

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.previous_error = error

        return output
