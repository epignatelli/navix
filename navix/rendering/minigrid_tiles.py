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


"""MiniGrid's tile renderer, as a table indexed by the symbolic encoding.

MiniGrid draws every tile procedurally: shapes are filled on a 3x
supersampled canvas, optionally highlighted, then averaged down to the
tile size (`Grid.render_tile`). Its first-person RGB view is therefore a
pure function of the symbolic view - each cell's `(type, colour, state)`,
whether it is visible, and whether the agent stands on it - so
`rgb_first_person` renders by indexing `tiles()` with
`symbolic_first_person`. The full view is the same function of the
symbolic grid, the agent's direction and the world-frame visibility
mask, which is how `rgb` renders.

The drawing functions below are MiniGrid's (`minigrid.utils.rendering`
and each `WorldObj.render`, Apache-2.0), kept scalar and in the same
float precision so the table is bit-identical to MiniGrid's tiles. It is
built once per tile size, with numpy, the first time it is asked for.
"""

from __future__ import annotations

import functools
import math

import numpy as np

# MiniGrid's OBJECT_TO_IDX, COLORS (in COLOR_TO_IDX order) and door states;
# the same values as `entities.EntityIds` and `rendering.registry.PALETTE`.
UNSEEN, EMPTY, WALL, DOOR, KEY, BALL, BOX, GOAL, LAVA, AGENT = (
    0, 1, 2, 4, 5, 6, 7, 8, 9, 10
)
NUM_TYPES, NUM_COLOURS, NUM_STATES, NUM_DIRECTIONS = 11, 6, 3, 4
COLOURS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [112, 39, 195],
        [255, 255, 0],
        [100, 100, 100],
    ]
)
DOOR_OPEN, DOOR_CLOSED, DOOR_LOCKED = 0, 1, 2
SUBDIVS = 3


def fill_coords(img, fn, color):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            yf = (y + 0.5) / img.shape[0]
            xf = (x + 0.5) / img.shape[1]
            if fn(xf, yf):
                img[y, x] = color
    return img


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy
        x2 = cx + x * math.cos(-theta) - y * math.sin(-theta)
        y2 = cy + y * math.cos(-theta) + x * math.sin(-theta)
        return fin(x2, y2)

    return fout


def point_in_line(x0, y0, x1, y1, r):
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    direction = p1 - p0
    dist = np.linalg.norm(direction)
    direction = direction / dist
    xmin, xmax = min(x0, x1) - r, max(x0, x1) + r
    ymin, ymax = min(y0, y1) - r, max(y0, y1) + r

    def fn(x, y):
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        q = np.array([x, y])
        a = np.clip(np.dot(q - p0, direction), 0, dist)
        return np.linalg.norm(q - (p0 + a * direction)) <= r

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax

    return fn


def point_in_triangle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    def fn(x, y):
        v0, v1, v2 = c - a, b - a, np.array((x, y)) - a
        dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
        dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn


def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    img[:, :, :] = blend_img.clip(0, 255).astype(np.uint8)


def downsample(img, factor):
    img = img.reshape(
        [img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3]
    )
    return img.mean(axis=3).mean(axis=1)


def draw_object(img, kind, colour, state):
    """`WorldObj.render` for the object encoded `(kind, colour, state)`."""
    c = COLOURS[colour]
    if kind in (WALL, GOAL):
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)
    elif kind == LAVA:
        fill_coords(img, point_in_rect(0, 1, 0, 1), (255, 128, 0))
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))
    elif kind == DOOR and state == DOOR_OPEN:
        fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
    elif kind == DOOR and state == DOOR_LOCKED:
        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))
        fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
    elif kind == DOOR:
        fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
        fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
        fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))
        fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)
    elif kind == KEY:
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))
    elif kind == BALL:
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), c)
    elif kind == BOX:
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)


def render_tile(canvas, agent, highlight):
    """`Grid.render_tile` over an object already drawn on `canvas`:
    `agent` is `0` for no agent and `1 + direction` for one facing
    MiniGrid's `direction`."""
    img = canvas.copy()
    if agent:
        triangle = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        triangle = rotate_fn(triangle, cx=0.5, cy=0.5, theta=0.5 * math.pi * (agent - 1))
        fill_coords(img, triangle, (255, 0, 0))
    if highlight:
        highlight_img(img)
    # MiniGrid writes the float mean into a uint8 frame, which truncates
    return downsample(img, SUBDIVS).astype(np.uint8)


def draw_cell(kind, colour, state, tile_size):
    """The grid lines and the object, on MiniGrid's supersampled canvas."""
    img = np.zeros((tile_size * SUBDIVS, tile_size * SUBDIVS, 3), dtype=np.uint8)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    draw_object(img, kind, colour, state)
    return img


@functools.lru_cache(maxsize=None)
def tiles(tile_size: int) -> np.ndarray:
    """Every tile MiniGrid can draw, indexed
    `[agent, highlight, type, colour, state]`.

    `agent` is `0` where the agent is not, and `1 + direction` where it
    stands facing `direction` (MiniGrid's: `0` east, clockwise), drawn
    over whatever shares its cell (`EMPTY` for nothing). `highlight` is
    MiniGrid's field-of-view tint. Each object is drawn once and the
    agent and tint laid over copies, which is the order `render_tile`
    applies them in. Encodings MiniGrid never produces, `UNSEEN`
    included, are left black.

    Args:
        tile_size (int): edge of one tile, in pixels.

    Returns:
        np.ndarray: `u8[1 + NUM_DIRECTIONS, 2, NUM_TYPES, NUM_COLOURS,
        NUM_STATES, tile_size, tile_size, 3]`."""
    table = np.zeros(
        (1 + NUM_DIRECTIONS, 2, NUM_TYPES, NUM_COLOURS, NUM_STATES)
        + (tile_size, tile_size, 3),
        dtype=np.uint8,
    )
    cells = [(EMPTY, 0, 0)]
    for kind in (WALL, GOAL, LAVA, KEY, BALL, BOX):
        cells += [(kind, colour, 0) for colour in range(NUM_COLOURS)]
    for colour in range(NUM_COLOURS):
        cells += [(DOOR, colour, state) for state in range(NUM_STATES)]
    for kind, colour, state in cells:
        canvas = draw_cell(kind, colour, state, tile_size)
        for agent in range(1 + NUM_DIRECTIONS):
            for highlight in (0, 1):
                table[agent, highlight, kind, colour, state] = render_tile(canvas, agent, highlight)
    return table
