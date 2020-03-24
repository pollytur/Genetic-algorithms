import datetime
import random
import numpy as np
from PIL import Image, ImageDraw

x_bound = 512
y_bound = 512
min_num_of_circles = 50
max_num_of_circles = 200
mutation_rate_min = 0.1
mutation_rate_max = 0.9
image_length = 512 * 512 * 3
population_size = 12
generations = 10
group_size = 4
min_radius = 5
max_radius = int(x_bound / 50)
to_delete_percentage = 0.1
# this % are relatively to the deleted number to slow down the growth of circles
to_add_min = 0.5
to_add_max = 1.5


class Circle:

    def __init__(self, origin):
        pos_x, pos_y = random.randint(0, x_bound-1), random.randint(0, y_bound-1)
        self.center = (pos_x, pos_y)
        self.radius = random.randint(min_radius, max_radius)
        col = origin[pos_x, pos_y, :]
        self.color = (col[0], col[1], col[2])


class Picture:
    # origin is np array of 512*512*3
    def __init__(self, origin):
        self.circles = []
        num_of_circles = random.randint(min_num_of_circles, max_num_of_circles)
        for _ in range(num_of_circles):
            self.circles.append(Circle(origin))
        self.fitness=0

    # transforms circles to np.array and calculates the MSE-fitness to the target picture
    def draw_circles_and_find_dist(self, origin):
        WHITE = (255, 255, 255)
        image = Image.new('RGB', (512, 512), WHITE)
        draw = ImageDraw.Draw(image)
        for x in self.circles:
            up_left_x = x.center[0] - x.radius
            up_left_y = x.center[1] - x.radius
            down_right_x = x.center[0] + x.radius
            down_right_y = x.center[1] + x.radius
            draw.ellipse([(int(up_left_x), int(up_left_y)), (int(down_right_x), int(down_right_y))], x.color)

        arr = np.asarray(image)
        self.fitness = (np.sum((origin - arr) ** 2))

    # changes the circles for a picture instance
    def set(self, new_circles, origin):
        self.circles = new_circles
        self.draw_circles_and_find_dist(origin)

    # checks that it should be at least 2 circles after mutation
    def mutate(self, origin):
        to_delete = random.randint(0, int(to_delete_percentage * len(self.circles)))
        to_del = random.sample(range(0, len(self.circles) - 1), to_delete)
        to_del.sort()
        to_del.reverse()
        [self.circles.pop(i) for i in to_del]
        to_add = random.randint(int(to_add_min * to_delete), int(to_add_max * to_delete))
        for i in range(to_add):
            self.circles.append(Circle(origin))
        self.draw_circles_and_find_dist(origin)

    # transforms Picture object to .jpg file
    def to_picture(self, name):
        WHITE = (255, 255, 255)
        image = Image.new('RGB', (x_bound, y_bound), WHITE)
        draw = ImageDraw.Draw(image)
        for x in self.circles:
            up_left_x = x.center[0] - x.radius
            up_left_y = x.center[1] - x.radius
            down_right_x = x.center[0] + x.radius
            down_right_y = x.center[1] + x.radius
            draw.ellipse([(int(up_left_x), int(up_left_y)), (int(down_right_x), int(down_right_y))], x.color)

        arr = np.asarray(image)
        final = Image.fromarray((arr * 255).astype(np.uint8))
        final.save(name + ".jpg")


def upload_picture(filename):
    target = Image.open(filename)
    return np.asarray(target)


def generate_population(origin):
    population = []
    for _ in range(population_size):
        p = Picture(origin)
        p.draw_circles_and_find_dist(origin)
        population.append(p)
    return population


def update_population(population, origin, i):
    population = increase_population(population, origin)
    population = decrease_population(population)
    best = select_best(population)
    best.to_picture(f"best_iter{i}")
    return population


def increase_population(population, origin):
    print("in increase_population ", datetime.datetime.now().time())
    random.shuffle(population)
    # divide on groups of size 4
    groups = []
    n = len(population)
    for i in range(0, n, group_size):
        groups.append(population[i:i + group_size])

    # for each group choose 2 most feet to be parents and produce kids
    for group in groups:
        if len(group) == 1:
            child = (group[0]).mutate(origin)
            population.append(child)
        else:
            # choose 2 most fit
            fitness = [gene.fitness for gene in group]
            idxs = np.argsort(fitness)
            idxs = idxs[-3:]
            parent1, parent2 = population[idxs[0]], population[idxs[1]]
            child = crossover(parent1, parent2, origin)
            child.mutate(origin)

            if child.fitness >= parent1.fitness or child.fitness >= parent2.fitness:
                population.append(child)

            if len(group) > 2:
                second_p = 2
            else:
                second_p = 1

            parent1, parent2 = population[idxs[0]], population[idxs[second_p]]
            child = crossover(parent1, parent2, origin)
            child.mutate(origin)

            if child.fitness >= parent1.fitness or child.fitness >= parent2.fitness:
                population.append(child)

    return population


def decrease_population(population):
    print("in decrease_population ", datetime.datetime.now().time())
    fitness = [gene.fitness for gene in population]
    cur_len = len(population)
    if cur_len > population_size:
        idxs = np.argsort(fitness)
        idxs = idxs[-population_size:]
        new_population = [population[i] for i in idxs]
        return new_population
    else:
        return population


def select_best(population):
    fitness = [gene.fitness for gene in population]
    idxs = np.argsort(fitness)
    return population[idxs[-1]]


# parent1 and parent2 are picture instances, origin is np.array 512*512*3
def crossover(parent1, parent2, origin):
    print("in crossover ", datetime.datetime.now().time())
    l1 = len(parent1.circles)
    l2 = len(parent2.circles)
    min_l = min(l1, l2)
    # parent one always have less or equal circles
    if len(parent2.circles) == min_l:
        p = parent1
        parent1 = parent2
        parent2 = p
    divisions = random.randint(2, round(min_l / 2))
    chunk_min = round(min_l / divisions)
    last_min = 0
    chunk_max = round(min_l / divisions)
    last_max = 0
    new_circles = []
    for i in range(divisions):
        if i == divisions - 1:
            if divisions % 2 == 0:
                new_circles.append(parent1.circles[last_min:])
            else:
                new_circles.append(parent2.circles[last_max:])
        else:
            if i % 2 == 0:
                new_circles.append(parent1.circles[last_min:last_min + chunk_min])
                last_min += chunk_min
            else:
                new_circles.append(parent2.circles[last_max:last_max + chunk_max])
                last_max += chunk_max

    new_circles_flattened = [item for sublist in new_circles for item in sublist]
    child = parent1
    child.set(new_circles_flattened, origin)

    return child

# to run the pipeline of the genetic algorithm call this function
# name is name of the target picture
def pipe(name):
    origin = upload_picture(name)
    population = generate_population(origin)
    for i in range(generations):
        population = update_population(population, origin, i)
        print(f"this is generation {i}")
    best = select_best(population)
    best.to_picture('final')


pipe()
