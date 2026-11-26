"""Training areas (row0, row1, col0, col1) in pixel coordinates of the 10 m grid.

Chosen by visual interpretation of the 2017 AND 2026 true-colour images, only where the
land cover is the same in both years ("temporally stable"), so the same areas can train
the classifier for every year.  0 = built-up, 1 = vegetation, 2 = water, 3 = bare / dry hills.
"""
CLASSES = {0: "Built-up", 1: "Vegetation", 2: "Water", 3: "Bare / dry hills"}
AREAS = {
    0: [(160, 290, 335, 410), (450, 550, 340, 420), (190, 320, 1270, 1370), (660, 790, 1430, 1550),
        (110, 175, 910, 1040), (830, 890, 1260, 1370)],
    1: [(1100, 1250, 30, 170), (600, 650, 245, 320), (530, 590, 910, 990)],
    2: [(300, 600, 30, 150), (470, 515, 625, 690), (1285, 1320, 10, 90)],
    3: [(110, 290, 1460, 1590), (60, 240, 650, 750), (1160, 1340, 1110, 1290), (1210, 1280, 755, 795),  # quarry (bright bare)
        (305, 375, 1570, 1615), (230, 280, 1660, 1710), (380, 440, 1680, 1760)],  # quarry, open bright, open dark land
}

# Year-specific areas on the airport site, where land cover changed (labelled from each year's image):
#   2017-18: wet scrub / mudflat (non-built); 2019-23: levelled earth fill (bare);
#   2024-26: unpaved infield (bare); 2025-26: runway pavement and terminal/apron buildings (built-up).
YEAR_AREAS = {}
for _y in (2017, 2018):
    YEAR_AREAS[_y] = {3: [(740, 800, 870, 950)]}
for _y in (2019, 2020, 2021, 2022, 2023):
    YEAR_AREAS[_y] = {3: [(700, 780, 950, 1100)]}
YEAR_AREAS[2024] = {3: [(700, 760, 950, 1050)]}
for _y in (2025, 2026):
    YEAR_AREAS[_y] = {3: [(700, 760, 950, 1050)], 0: [(818, 842, 850, 1080), (775, 805, 1110, 1150)]}
