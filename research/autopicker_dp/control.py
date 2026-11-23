"""Joint commands in radians/meters, independent of Isaac Sim."""
import numpy as np


def vector(value, size=4):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {size} finite values')
    return result.copy()


class JointExecutor:
    def __init__(self, lower, upper, velocity, acceleration, dt):
        self.lower, self.upper = vector(lower), vector(upper)
        self.max_velocity, self.max_acceleration = vector(velocity), vector(acceleration)
        self.dt = float(dt)
        if (not np.isfinite(self.dt) or self.dt <= 0 or np.any(self.lower >= self.upper)
                or np.any(self.max_velocity <= 0) or np.any(self.max_acceleration <= 0)):
            raise ValueError('Invalid joint limits or timestep')
        self.position = None
        self.velocity = np.zeros(4)
        self.limited_steps = 0

    def reset(self, position):
        position = vector(position)
        if np.any(position < self.lower) or np.any(position > self.upper):
            raise ValueError('Initial position outside joint limits')
        self.position = position
        self.velocity = np.zeros(4)
        self.limited_steps = 0

    def advance(self, requested):
        requested = vector(requested)
        if self.position is None:
            raise RuntimeError('Reset the executor before commanding motion')
        if np.any(requested < self.lower) or np.any(requested > self.upper):
            raise ValueError('Requested position outside joint limits')
        error = requested - self.position
        braking_step = self.max_acceleration * self.dt
        stopping_speed = np.sqrt(braking_step**2 + 2 * self.max_acceleration * np.abs(error)) - braking_step
        desired = np.sign(error) * np.minimum(np.abs(error) / self.dt,
                                               np.minimum(self.max_velocity, stopping_speed))
        velocity = np.clip(desired, self.velocity - self.max_acceleration * self.dt,
                           self.velocity + self.max_acceleration * self.dt)
        position = self.position + velocity * self.dt
        if np.any(position < self.lower - 1e-9) or np.any(position > self.upper + 1e-9):
            raise ValueError('Command would cross a joint limit; reset or brake before the boundary')
        self.limited_steps += int(np.max(np.abs(position - requested)) > 1e-8)
        self.position, self.velocity = position, velocity
        return position.copy(), velocity.copy()


class CommandTrajectory:
    def __init__(self, times, commands):
        self.times = np.asarray(times, dtype=np.float64)
        self.commands = np.asarray(commands, dtype=np.float64)
        if (self.times.ndim != 1 or len(self.times) < 2
                or self.commands.shape != (len(self.times), 4)
                or not np.isfinite(self.times).all() or not np.isfinite(self.commands).all()
                or np.any(np.diff(self.times) <= 0)):
            raise ValueError('Expected increasing finite times and Nx4 commands')

    def at(self, time_s):
        if not np.isfinite(time_s) or not self.times[0] - 1e-9 <= time_s <= self.times[-1] + 1e-9:
            raise ValueError('Time is outside the recorded command trajectory')
        return np.array([np.interp(time_s, self.times, self.commands[:, index]) for index in range(4)])

    def resample(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Sampling interval must be positive')
        times = np.arange(self.times[0], self.times[-1], dt)
        times = np.append(times, self.times[-1])
        return CommandTrajectory(times, np.asarray([self.at(time_s) for time_s in times]))
