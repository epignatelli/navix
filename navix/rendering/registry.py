# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations

import os
from PIL import Image

import jax
from jax import Array
import jax.numpy as jnp


SPRITES_DIR = os.path.normpath(
    os.path.join(__file__, "..", "..", "..", "assets", "sprites")
)
MIN_TILE_SIZE = 8
TILE_SIZE = MIN_TILE_SIZE


def load_sprite(name: str) -> Array:
    """Loads an image from disk in RGB space.
    Args:
        path(str): the filepath of the image on disk

    Returns:
        (Array): a jax.Array of shape (H, W, C)"""
    path = os.path.join(SPRITES_DIR, f"{name}.png")
    image = Image.open(path)
    array = jnp.asarray(image)
    resized = jax.image.resize(array, (TILE_SIZE, TILE_SIZE, 3), method="cubic")
    return jnp.asarray(resized, dtype=jnp.uint8)


class PALETTE:
    RED: Array = jnp.asarray(0, dtype=jnp.uint8)
    GREEN: Array = jnp.asarray(1, dtype=jnp.uint8)
    BLUE: Array = jnp.asarray(2, dtype=jnp.uint8)
    PURPLE: Array = jnp.asarray(3, dtype=jnp.uint8)
    YELLOW: Array = jnp.asarray(4, dtype=jnp.uint8)
    GREY: Array = jnp.asarray(5, dtype=jnp.uint8)

    @classmethod
    def as_string(cls):
        return ["red", "green", "blue", "purple", "yellow", "grey"]

    @classmethod
    def as_array(cls):
        return [cls.RED, cls.GREEN, cls.BLUE, cls.PURPLE, cls.YELLOW, cls.GREY]


class SpritesRegistry:
    def __init__(self):
        self.registry = {}
        self.build_registry()

    def build_registry(self):
        """Populates the sprites registry for all entities."""
        self.set_wall_sprite()
        self.set_floor_sprite()
        self.set_goal_sprite()
        self.set_key_sprite()
        self.set_player_sprite()
        self.set_door_sprite()
        self.set_lava_sprite()
        self.set_ball_sprite()
        self.set_box_sprite()

    def set_wall_sprite(self):
        self.registry["wall"] = load_sprite("wall")

    def set_floor_sprite(self):
        self.registry["floor"] = load_sprite("floor")

    def set_goal_sprite(self):
        self.registry["goal"] = load_sprite("goal")

    def set_key_sprite(self):
        keys_coloured = [
            load_sprite("key" + f"_{colour}") for colour in PALETTE.as_string()
        ]
        self.registry["key"] = jnp.stack(keys_coloured, axis=0)

    def set_player_sprite(self):
        self.registry["player"] = jnp.stack(
            [
                load_sprite("player_east"),
                load_sprite("player_south"),
                load_sprite("player_west"),
                load_sprite("player_north"),
            ]
        )

    def set_door_sprite(self):
        door = jnp.zeros(
            (len(PALETTE.as_string()), 3, TILE_SIZE, TILE_SIZE, 3), dtype=jnp.uint8
        )
        for c_idx, colour in enumerate(PALETTE.as_string()):
            for s_idx, state in enumerate(["closed", "open", "locked"]):
                sprite = load_sprite("door" + f"_{state}" + f"_{colour}")
                door = door.at[c_idx, s_idx].set(sprite)
        self.registry["door"] = door

    def set_lava_sprite(self):
        self.registry["lava"] = load_sprite("lava")

    def set_ball_sprite(self):
        ball_coloured = [
            load_sprite("ball" + f"_{colour}") for colour in PALETTE.as_string()
        ]
        self.registry["ball"] = jnp.stack(ball_coloured, axis=0)

    def set_box_sprite(self):
        box_coloured = [
            load_sprite("box" + f"_{colour}") for colour in PALETTE.as_string()
        ]
        self.registry["box"] = jnp.stack(box_coloured, axis=0)


# initialise sprites registry
SPRITES_REGISTRY = SpritesRegistry().registry
