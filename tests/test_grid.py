import navix as nx


def test_grid():
    ascii = \
    """########
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
    print(ascii)

    grid = nx.grid.from_ascii(ascii)
    print(grid)

    ascii = ascii.replace("1", "P")
    ascii = ascii.replace("2", "G")
    grid = nx.grid.from_ascii(ascii, mapping={"P": 1, "G": 2})
    print(grid)


if __name__ == "__main__":
    test_grid()