import navix as nx


def test_grid():
    ascii_map = """########
    #1.....#
    #......#
        #......#
#......#
        #......#
    #......#
    #.....2#
########
########
########
########
    """
    print(ascii_map)

    grid = nx.grid.from_ascii_map(ascii_map)
    print(grid)

    ascii_map = ascii_map.replace("1", "P")
    ascii_map = ascii_map.replace("2", "G")
    grid = nx.grid.from_ascii_map(ascii_map, mapping={"P": 1, "G": 2})
    print(grid)


# if __name__ == "__main__":
#     test_grid()
