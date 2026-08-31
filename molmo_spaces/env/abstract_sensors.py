# rmh: pulled from allenact and dependencies stripped out
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    Any,
)

import gymnasium as gym
import gymnasium.spaces as gyms

SpaceDict = gyms.Dict


class Sensor(ABC):
    """Represents a sensor that provides data from the environment to agent.
    The user of this class needs to implement the get_observation method and
    the user is also required to set the below attributes:

    # Attributes

    uuid : universally unique id.
    observation_space : ``gym.Space`` object corresponding to observation of
        sensor.
    is_dict : whether the observation is a dictionary
    str_max_len : maximum length of the string representation of the encoded dictionary, if is_dict is True
    """

    uuid: str
    observation_space: gym.Space
    is_dict: bool = False
    str_max_len: int = 2000

    def __init__(self, uuid: str, observation_space: gym.Space, **kwargs: Any) -> None:
        self.uuid = uuid
        self.observation_space = observation_space

    @abstractmethod
    def get_observation(self, env, task, *args: Any, **kwargs: Any) -> Any:
        """Returns observations from the environment (or task).

        # Parameters

        env : The environment the sensor is used upon.
        task : (Optionally) a Task from which the sensor should get data.

        # Returns

        Current observation for Sensor.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Reset the sensor to its initial state."""
        return None


class SensorSuite:
    """Represents a set of sensors, with each sensor being identified through a
    unique id.

    # Attributes

    sensors: list containing sensors for the environment, uuid of each
        sensor must be unique.
    """

    sensors: dict[str, Sensor]
    observation_spaces: gyms.Dict

    def __init__(self, sensors: Sequence[Sensor]) -> None:
        """Initializer.

        # Parameters

        param sensors: the sensors that will be included in the suite.
        """
        self.sensors = OrderedDict()
        spaces: OrderedDict[str, gym.Space] = OrderedDict()
        for sensor in sensors:
            assert sensor.uuid not in self.sensors, f"'{sensor.uuid}' is duplicated sensor uuid"
            self.sensors[sensor.uuid] = sensor
            spaces[sensor.uuid] = sensor.observation_space
        self.observation_spaces = SpaceDict(spaces=spaces)

    def get(self, uuid: str) -> Sensor:
        """Return sensor with the given `uuid`.

        # Parameters

        uuid : The unique id of the sensor

        # Returns

        The sensor with unique id `uuid`.
        """
        return self.sensors[uuid]

    def get_observations(self, env, task, **kwargs: Any) -> dict[str, Any]:
        """Get all observations corresponding to the sensors in the suite.

        # Parameters

        env : The environment from which to get the observation.
        task : (Optionally) the task from which to get the observation.

        # Returns

        Data from all sensors packaged inside a Dict.
        """
        return {
            uuid: sensor.get_observation(env=env, task=task, **kwargs)  # type: ignore
            for uuid, sensor in self.sensors.items()
        }

    def render_requests(self) -> list[tuple[str, str]]:
        """(camera_name, mode) pairs this suite needs rendered, deduplicated
        -- mode is one of "rgb"/"depth"/"segmentation", matching
        CPUMujocoEnv.render_batch's `mode` parameter. Used by
        BaseMujocoTask.get_observations() to batch-render a whole group of
        indices' cameras together instead of one call at a time.

        Covers every sensor that ever triggers a render, directly or not:
        CameraSensor/DepthSensor/SegmentationSensor call env.render_*_frame
        directly; ObjectImagePointsSensor calls
        env.get_segmentation_mask_of_object(), which internally calls
        env.render_segmentation_frame() -- same render, same cache
        (render_rgb_frame/render_depth_frame/render_segmentation_frame all
        check CPUMujocoEnv._render_cache before rendering), so pre-warming it
        here makes that call a cache hit too without ObjectImagePointsSensor
        needing to know anything about batching."""
        from molmo_spaces.env.sensors import ObjectImagePointsSensor
        from molmo_spaces.env.sensors_cameras import CameraSensor, DepthSensor, SegmentationSensor

        seen: set[tuple[str, str]] = set()
        requests: list[tuple[str, str]] = []

        def _add(key: tuple[str, str]) -> None:
            if key not in seen:
                seen.add(key)
                requests.append(key)

        for sensor in self.sensors.values():
            if isinstance(sensor, CameraSensor):
                _add((sensor.camera_name, "rgb"))
            elif isinstance(sensor, DepthSensor):
                _add((sensor.camera_name, "depth"))
            elif isinstance(sensor, SegmentationSensor):
                _add((sensor.camera_name, "segmentation"))
            elif isinstance(sensor, ObjectImagePointsSensor):
                for camera_name in sensor.camera_names:
                    _add((camera_name, "segmentation"))
        return requests

    def extend(self, sensors: Sequence[Sensor]) -> None:
        """Extend the sensor suite with additional sensors."""
        for sensor in sensors:
            self.add(sensor)

    def add(self, sensor: Sensor) -> None:
        """Add a sensor to the sensor suite."""
        assert sensor.uuid not in self.sensors, f"'{sensor.uuid}' is duplicated sensor uuid"
        self.sensors[sensor.uuid] = sensor
        self.observation_spaces[sensor.uuid] = sensor.observation_space
