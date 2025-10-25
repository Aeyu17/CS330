# Jamie Roberson for CS330 FA-25
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from typing import Optional # Optionals used because Python 3.9 does not have built-in support for union types of aggregate types (i.e. tuples)
# for non-aggregate types, just used the built-in Union type

from utils import set_magnitude, get_length, dot

class SteeringBehavior(Enum):
    CONTINUE = 1
    SEEK = 6
    FLEE = 7
    ARRIVE = 8
    FOLLOW_PATH = 11

@dataclass
class Timestep:
    sim_time: float
    character_id: int
    position: tuple[float, float] # (x, z)
    velocity: tuple[float, float] # (vx, vz)
    linear_acceleration: tuple[float, float] # (ax, az)
    orientation: float # radians
    steering_behavior: SteeringBehavior # 1=Continue, 6=Seek, 7=Flee, 8=Arrive, 11=Follow Path
    collision: bool

    # Used for writing to file
    def to_string(self) -> str:
        return f"{self.sim_time},{self.character_id},{self.position[0]},{self.position[1]},{self.velocity[0]},{self.velocity[1]},{self.linear_acceleration[0]},{self.linear_acceleration[1]},{self.orientation},{self.steering_behavior.value},{self.collision}"

class Movement:
    instance_number = 1
    def __init__(self, id=0, behavior=SteeringBehavior.CONTINUE, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0), init_orient=0.0, max_vel=0.0, max_acc=0.0, target_instance=0, arrival_radius=None, slow_radius=None, time_to_target=None, path_to_follow=None, path_offset=None):
        
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
        self.path_to_follow: Path | None = path_to_follow
        self.path_offset: float | None = path_offset


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
        
        elif self.behavior == SteeringBehavior.FOLLOW_PATH:
            curr_param = self.path_to_follow.get_path_parameter(self.position)
            target_param = curr_param + self.path_offset
            path_target_pos = self.path_to_follow.get_closest_position(target_param)

            # Seek behaviour
            return set_magnitude((path_target_pos[0] - self.position[0], path_target_pos[1] - self.position[1]), self.max_acceleration)
        
        else:
            raise ValueError("Invalid Steering Behavior")
        
class Path:
    instance_number = 1
    def __init__(self, waypoints: list[tuple[float, float]]):
        # Unique instance number for each Path object
        self.instance_number = Path.instance_number
        Path.instance_number += 1
        self.waypoints = waypoints

        if len(self.waypoints) == 0:
            raise ValueError

        # Assemble path
        segments = len(self.waypoints) - 1
        self.distance = [0]
        for i in range(1, segments+1):
            # Store total distance travelled between waypoints
            self.distance.append(self.distance[i-1] + get_length((self.waypoints[i][0] - self.waypoints[i-1][0], self.waypoints[i][1] - self.waypoints[i-1][1])))
        self.path_params = [0]
        for i in range(1, segments+1):
            # Normalise the distance to get "path parameter" between 0 and 1
            self.path_params.append(self.distance[i] / max(self.distance))
    
    def get_closest_position(self, curr_path_param: float) -> tuple[float, float]:
        idx = self.path_params.index(max([i for i in self.path_params if i <= curr_path_param]))
        A = self.waypoints[idx]
        B = self.waypoints[idx+1]
        # t parameterises the segment between A and B
        t = (curr_path_param - self.path_params[idx]) / (self.path_params[idx+1] - self.path_params[idx])
        return (A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t)
    
    def get_path_parameter(self, curr_pos: tuple[float, float]) -> float:
        closest_distance = math.inf
        closest_index = None
        closest_point = None

        # Find the closest point on the path to the current position
        for i in range(len(self.waypoints)-1):
            A = self.waypoints[i]
            B = self.waypoints[i+1]
            candidate_point = closest_point_on_path(curr_pos, A, B)
            dist = get_length((candidate_point[0] - curr_pos[0], candidate_point[1] - curr_pos[1]))
            
            if closest_distance > dist:
                closest_distance = dist
                closest_index = i
                closest_point = candidate_point

        if closest_index is None or closest_point is None:
            raise ValueError
        
        A = self.waypoints[closest_index]
        A_param = self.path_params[closest_index]
        B = self.waypoints[closest_index+1]
        B_param = self.path_params[closest_index+1]

        t = get_length((closest_point[0] - A[0], closest_point[1] - A[1])) / get_length((B[0] - A[0], B[1] - A[1]))
        return (A_param + (B_param - A_param) * t) # Get the path parameter at the closest point
                
    
def get_target_position(target_instance: int, movement_list: list[Movement]) -> Optional[tuple[float, float]]:
    # Helper function to find target position for updating timestep
    for movement in movement_list:
        if movement.instance_number == target_instance:
            return movement.position
    return None # did not find target

def closest_point_on_path(curr_pos: tuple[float, float], A: tuple[float, float], B: tuple[float, float]) -> tuple[float, float]:
    diff_A = (curr_pos[0] - A[0], curr_pos[1] - A[1])
    diff_B = (B[0] - A[0], B[1] - A[1])
    t = dot(diff_A, diff_B) / dot(diff_B, diff_B)
    if t < 0.0:
        return A
    elif t > 1.0:
        return B
    else:
        return (A[0] + diff_B[0] * t, A[1] + diff_B[1] * t)

def main():
    path = Path(waypoints=[(0, 90), (-20, 65), (20, 40), (-40, 15), (40, -10), (-60, -35), (60, -60), (0, -85)])
    # Initialise characters based on assignment specs
    follow_path = Movement(id=2701,
                           behavior=SteeringBehavior.FOLLOW_PATH,
                           init_pos=(20.0, 95.0),
                           init_vel=(0.0, 0.0),
                           init_orient=0.0,
                           max_vel=4.0,
                           max_acc=2.0,
                           path_to_follow=path,
                           path_offset=0.04)
    
    character_list = [follow_path]

    data_filename = "data.txt"

    with open(data_filename, 'w') as _:
        pass # clear file at start of program

    for i in np.linspace(0, 125, num=125*2+1):
        dt = 0.5
        for j, character in enumerate(character_list):
            print(f"Updating character {j} at time {i}")
            timestep: Timestep = character.update_timestep(i, dt, get_target_position(character.target_id, character_list))
            with open(data_filename, 'a') as f:
                f.write(timestep.to_string() + "\n")
                # print(timestep.to_string()) # Debug print statement to see timesteps


if __name__ == "__main__":
    main()