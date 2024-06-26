# Supported environments 

NAVIX is designed to be a drop-in replacement for the official MiniGrid environments.
You can reuse your existing code and scripts with NAVIX with little to no modification.

You can find the original MiniGrid environments in the [MiniGrid documentation](https://minigrid.huggingface.co/docs/).
For more details on MiniGrid, have a look also at the [original publication](https://arxiv.org/pdf/2306.13831).

The following table lists the supported MiniGrid environments and their corresponding NAVIX environments.
If you cannot find the environment you are looking for, please consider [opening a feature request](https://github.com/epignatelli/navix/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=) on GitHub.

| MiniGrid ID                                  | NAVIX ID                                                                              | Description                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `MiniGrid-Empty-5x5-v0`                      | [`Navix-Empty-5x5-v0`](../api/environments/empty.md)                                  | Empty 5x5 grid                                                           |
| `MiniGrid-Empty-6x6-v0`                      | [`Navix-Empty-6x6-v0`](../api/environments/empty.md)                                  | Empty 6x6 grid                                                           |
| `MiniGrid-Empty-8x8-v0`                      | [`Navix-Empty-8x8-v0`](../api/environments/empty.md)                                  | Empty 8x8 grid                                                           |
| `MiniGrid-Empty-16x16-v0`                    | [`Navix-Empty-16x16-v0`](../api/environments/empty.md)                                | Empty 16x16 grid                                                         |
| `MiniGrid-Empty-Random-5x5-v0`               | [`Navix-Empty-Random-5x5-v0`](../api/environments/empty.md)                           | Empty 5x5 grid with random starts                                        |
| `MiniGrid-Empty-Random-6x6-v0`               | [`Navix-Empty-Random-6x6-v0`](../api/environments/empty.md)                           | Empty 6x6 grid with random starts                                        |
| `MiniGrid-Empty-Random-8x8-v0`               | [`Navix-Empty-Random-8x8-v0`](../api/environments/empty.md)                           | Empty 8x8 grid with random starts                                        |
| `MiniGrid-Empty-Random-16x16-v0`             | [`Navix-Empty-Random-16x16-v0`](../api/environments/empty.md)                         | Empty 16x16 grid with random starts                                      |
| `MiniGrid-FourRooms-v0`                      | [`Navix-FourRooms-v0`](../api/environments/four_rooms.md)                             | Four rooms                                                               |
| `MiniGrid-DoorKey-5x5-v0`                    | [`Navix-DoorKey-5x5-v0`](../api/environments/door_key.md)                             | 5x5 grid with a key and a door                                           |
| `MiniGrid-DoorKey-6x6-v0`                    | [`Navix-DoorKey-6x6-v0`](../api/environments/door_key.md)                             | 6x6 grid with a key and a door                                           |
| `MiniGrid-DoorKey-8x8-v0`                    | [`Navix-DoorKey-8x8-v0`](../api/environments/door_key.md)                             | 8x8 grid with a key and a door                                           |
| `MiniGrid-DoorKey-16x16-v0`                  | [`Navix-DoorKey-16x16-v0`](../api/environments/door_key.md)                           | 16x16 grid with a key and a door                                         |
| `MiniGrid-DoorKey-5x5-Random-v0`             | [`Navix-DoorKey-5x5-Random-v0`](../api/environments/door_key.md)                      | 5x5 grid with a key and a door                                           |
| `MiniGrid-DoorKey-6x6-Random-v0`             | [`Navix-DoorKey-6x6-Random-v0`](../api/environments/door_key.md)                      | 6x6 grid with a key and a door                                           |
| `MiniGrid-DoorKey-8x8-Random-v0`             | [`Navix-DoorKey-8x8-Random-v0`](../api/environments/door_key.md)                      | 8x8 grid with a key and a door                                           |
| `MiniGrid-DoorKey-16x16-Random-v0`           | [`Navix-DoorKey-16x16-Random-v0`](../api/environments/door_key.md)                    | 16x16 grid with a key and a door                                         |
| `MiniGrid-KeyCorridorS3R1-v0`                | [`Navix-KeyCorridorS3R1-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 3 cells away                                         |
| `MiniGrid-KeyCorridorS3R2-v0`                | [`Navix-KeyCorridorS3R2-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 3 cells away                                         |
| `MiniGrid-KeyCorridorS3R3-v0`                | [`Navix-KeyCorridorS3R3-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 3 cells away                                         |
| `MiniGrid-KeyCorridorS4R3-v0`                | [`Navix-KeyCorridorS4R3-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 4 cells away                                         |
| `MiniGrid-KeyCorridorS5R3-v0`                | [`Navix-KeyCorridorS5R3-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 5 cells away                                         |
| `MiniGrid-KeyCorridorS6R3-v0`                | [`Navix-KeyCorridorS6R3-v0`](../api/environments/key_corridor.md)                     | Corridor with a key 6 cells away                                         |
| `MiniGrid-Crossings-S9N1-v0`                 | [`Navix-Crossings-S9N1-v0`](../api/environments/crossings.md)                         | A 9x9 room with 1 wall crossing it                                       |
| `MiniGrid-Crossings-S9N2-v0`                 | [`Navix-Crossings-S9N2-v0`](../api/environments/crossings.md)                         | A 9x9 room with 2 walls crossing it                                      |
| `MiniGrid-Crossings-S9N3-v0`                 | [`Navix-Crossings-S9N3-v0`](../api/environments/crossings.md)                         | A 9x9 room with 3 walls crossing it                                      |
| `MiniGrid-Crossings-S11N5-v0`                | [`Navix-Crossings-S11N5-v0`](../api/environments/crossings.md)                        | A 11x11 room with 5 walls crossing it                                    |
| `MiniGrid-DistShift1-v0`                     | [`Navix-DistShift1-v0`](../api/environments/dist_shift.md)                            | DistShift with 1 goal                                                    |
| `MiniGrid-DistShift2-v0`                     | [`Navix-DistShift2-v0`](../api/environments/dist_shift.md)                            | DistShift with 2 goals                                                   |
| `MiniGrid-LavaGap-S5-v0`                     | [`Navix-LavaGap-S5-v0`](../api/environments/lava_gap.md)                              | LavaGap with in a 5x5 room                                               |
| `MiniGrid-LavaGap-S6-v0`                     | [`Navix-LavaGap-S6-v0`](../api/environments/lava_gap.md)                              | LavaGap with in a 6x6 room                                               |
| `MiniGrid-LavaGap-S7-v0`                     | [`Navix-LavaGap-S7-v0`](../api/environments/lava_gap.md)                              | LavaGap with 7x7 room                                                    |
| `MiniGrid-GoToDoor-5x5-v0`                   | [`Navix-GoToDoor-5x5-v0`](../api/environments/go_to_door.md)                          | 5x5 grid that terminates with a `done` action next to a certain door     |
| `MiniGrid-GoToDoor-6x6-v0`                   | [`Navix-GoToDoor-6x6-v0`](../api/environments/go_to_door.md)                          | 6x6 grid that terminates with a `done` action next to a certain doo      |
| `MiniGrid-GoToDoor-8x8-v0`                   | [`Navix-GoToDoor-8x8-v0`](../api/environments/go_to_door.md)                          | 8x8 grid grid that terminates with a `done` action next to a certain doo |
| `MiniGrid-Dynamic-Obstacles-5x5-v0`          | [`Navix-Dynamic-Obstacles-5x5-v0`](../api/environments/dynamic_obstacles.md)          | 5x5 grid with dynamic obstacles                                          |
| `MiniGrid-Dynamic-Obstacles-6x6-v0`          | [`Navix-Dynamic-Obstacles-6x6-v0`](../api/environments/dynamic_obstacles.md)          | 6x6 grid with dynamic obstacles                                          |
| `MiniGrid-Dynamic-Obstacles-8x8-v0`          | [`Navix-Dynamic-Obstacles-8x8-v0`](../api/environments/dynamic_obstacles.md)          | 8x8 grid with dynamic obstacles                                          |
| `MiniGrid-Dynamic-Obstacles-16x16-v0`        | [`Navix-Dynamic-Obstacles-16x16-v0`](../api/environments/dynamic_obstacles.md)        | 16x16 grid with dynamic obstacles                                        |
| `MiniGrid-Dynamic-Obstacles-Random-5x5-v0`   | [`Navix-Dynamic-Obstacles-Random-5x5-v0`](../api/environments/dynamic_obstacles.md)   | 5x5 grid with dynamic obstacles and random starts                        |
| `MiniGrid-Dynamic-Obstacles-Random-6x6-v0`   | [`Navix-Dynamic-Obstacles-Random-6x6-v0`](../api/environments/dynamic_obstacles.md)   | 6x6 grid with dynamic obstacles and random starts                        |
| `MiniGrid-Dynamic-Obstacles-Random-8x8-v0`   | [`Navix-Dynamic-Obstacles-Random-8x8-v0`](../api/environments/dynamic_obstacles.md)   | 8x8 grid with dynamic obstacles and random starts                        |
| `MiniGrid-Dynamic-Obstacles-Random-16x16-v0` | [`Navix-Dynamic-Obstacles-Random-16x16-v0`](../api/environments/dynamic_obstacles.md) | 16x16 grid with dynamic obstacles and random starts                      |
