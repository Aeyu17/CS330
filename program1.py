# Jamie Roberson for CS330 FA-25
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from typing import Optional # Optionals used because Python 3.9 does not have built-in support for union types of aggregate types (i.e. tuples)
# for non-aggregate types, just used the built-in Union type

from utils import set_magnitude, get_length

class SteeringBehavior(Enum):
    CONTINUE = 1
    SEEK = 6
    FLEE = 7
    ARRIVE = 8

@dataclass
class Timestep:
    sim_time: float
    character_id: int
    position: tuple[float, float] # (x, z)
    velocity: tuple[float, float] # (vx, vz)
    linear_acceleration: tuple[float, float] # (ax, az)
    orientation: float # radians
    steering_behavior: SteeringBehavior # 1=Continue, 6=Seek, 7=Flee, 8=Arrive
    collision: bool

    # Used for writing to file
    def to_string(self) -> str:
        return f"{self.sim_time},{self.character_id},{self.position[0]},{self.position[1]},{self.velocity[0]},{self.velocity[1]},{self.linear_acceleration[0]},{self.linear_acceleration[1]},{self.orientation},{self.steering_behavior.value},{int(self.collision)}"

class Movement:
    instance_number = 1
    def __init__(self, id=0, behavior=SteeringBehavior.CONTINUE, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0), init_orient=0.0, max_vel=0.0, max_acc=0.0, target_instance=0, arrival_radius=None, slow_radius=None, time_to_target=None):
        
        # Unique instance number for each Movement object
        self.instance_number = Movement.instance_number
        Movement.instance_number += 1

        self.character_id: int = id
        self.behavior: SteeringBehavior = behavior
        self.position: tuple[float, float] = init_pos
        self.velocity: tuple[float, float] = init_vel
        self.orientation: float = init_orient
        self.max_velocity: float = max_vel
        self.max_acceleration: float = max_acc
        self.target_id: int = target_instance

        # Optional parameters, not needed for all behaviors
        self.arrival_radius: float | None = arrival_radius
        self.slow_radius: float | None = slow_radius
        self.time_to_target: float | None = time_to_target

    def update_timestep(self, current_time: float, dt: float = 0.5, target_pos: Optional[tuple[float, float]] = None) -> Timestep:
        # Create Timestep object with current values, position, velocity, and linear acceleration will be updated
        returned_timestep = Timestep(sim_time=current_time,
                                     character_id=self.character_id,
                                     position=self.position,
                                     velocity=self.velocity,
                                     linear_acceleration=(0.0, 0.0),
                                     orientation=self.orientation,
                                     steering_behavior=self.behavior,
                                     collision=False)
        
        # Calculate acceleration based on behavior
        linear_acceleration = self.compute_acceleration(target_pos)
        
        # Take derivatives to update position and velocity
        self.position = (self.position[0] + self.velocity[0] * dt, self.position[1] + self.velocity[1] * dt)
        self.velocity = (self.velocity[0] + linear_acceleration[0] * dt, self.velocity[1] + linear_acceleration[1] * dt)

        # Check if velocity is too fast
        if get_length(self.velocity) > self.max_velocity:
            self.velocity = set_magnitude(self.velocity, self.max_velocity)

        returned_timestep.position = self.position
        returned_timestep.velocity = self.velocity
        returned_timestep.linear_acceleration = linear_acceleration

        # Return Timestep object with updated values
        return returned_timestep
    
    def compute_acceleration(self, target_pos: Optional[tuple[float, float]] = None) -> tuple[float, float]:
        if self.behavior == SteeringBehavior.CONTINUE:
            return (0.0, 0.0)

        elif self.behavior == SteeringBehavior.SEEK:
            return set_magnitude((target_pos[0] - self.position[0], target_pos[1] - self.position[1]), self.max_acceleration)
       
        elif self.behavior == SteeringBehavior.FLEE:
            return set_magnitude((self.position[0] - target_pos[0], self.position[1] - target_pos[1]), self.max_acceleration)
       
        elif self.behavior == SteeringBehavior.ARRIVE:
            direction = (target_pos[0] - self.position[0], target_pos[1] - self.position[1])
            distance = get_length(direction)

            # Arrived
            if distance < self.arrival_radius:
                return (0.0, 0.0)

            # Outside slow radius, go max acceleration
            if distance > self.slow_radius:
                target_speed = self.max_velocity

            # Inside slow radius
            else:
                target_speed = self.max_velocity * distance / self.slow_radius

            target_velocity = set_magnitude(direction, target_speed)
            linear_acceleration = ((target_velocity[0] - self.velocity[0]) / self.time_to_target, 
                                    (target_velocity[1] - self.velocity[1]) / self.time_to_target)

            # Check if acceleration is too fast
            if get_length(linear_acceleration) > self.max_acceleration:
                return set_magnitude(linear_acceleration, self.max_acceleration)
            
            return linear_acceleration
        
        else:
            raise ValueError("Invalid Steering Behavior")
        
    
def get_target_position(target_instance: int, movement_list: list[Movement]) -> Optional[tuple[float, float]]:
    # Helper function to find target position for updating timestep
    for movement in movement_list:
        if movement.instance_number == target_instance:
            return movement.position
    return None # did not find target
    

def main():
    # Initialise characters based on assignment specs
    continue_character = Movement(id=2601, 
                                  behavior=SteeringBehavior.CONTINUE, 
                                  init_pos=(0.0, 0.0), 
                                  init_vel=(0.0, 0.0), 
                                  max_acc=(0.0, 0.0), 
                                  init_orient=0.0)
    
    flee_character = Movement(id=2602, 
                              behavior=SteeringBehavior.FLEE, 
                              init_pos=(-30.0, -50.0), 
                              init_vel=(2.0, 7.0), 
                              init_orient=math.pi / 4,
                              max_vel=8.0,
                              max_acc=1.5,
                              target_instance=1)
    
    seek_character = Movement(id=2603, 
                              behavior=SteeringBehavior.SEEK, 
                              init_pos=(-50.0, 40.0), 
                              init_vel=(0.0, 8.0), 
                              init_orient=3 * math.pi / 2,
                              max_vel=8.0,
                              max_acc=2.0,
                              target_instance=1)
    
    arrive_character = Movement(id=2604, 
                                behavior=SteeringBehavior.ARRIVE, 
                                init_pos=(50.0, 75.0), 
                                init_vel=(-9.0, 4.0), 
                                init_orient=math.pi,
                                max_vel=10.0,
                                max_acc=2.0,
                                target_instance=1,
                                arrival_radius=4.0,
                                slow_radius=32.0,
                                time_to_target=1.0)
    
    character_list = [continue_character, flee_character, seek_character, arrive_character]

    data_filename = "data.txt"

    with open(data_filename, 'w') as _:
        pass # clear file at start of program

    for i in np.linspace(0, 50, num=101):
        dt = 0.5
        for j, character in enumerate(character_list):
            print(f"Updating character {j} at time {i}")
            timestep: Timestep = character.update_timestep(i, dt, get_target_position(character.target_id, character_list))
            with open(data_filename, 'a') as f:
                f.write(timestep.to_string() + "\n")
                # print(timestep.to_string()) # Debug print statement to see timesteps


if __name__ == "__main__":
    main()