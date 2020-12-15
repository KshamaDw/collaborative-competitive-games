import math
import gym
import time
from enum import IntEnum
from gym import error, spaces, utils
from gym.utils import seeding
# from .rendering import *
# from .window import Window
import numpy as np
from termcolor import colored,cprint
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = "Comic Sans MS"
plt.rcParams['font.serif'] = 'Times New Roman'

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'white': np.array([255,255,255]),
    'black': np.array([0,0,0])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

################################# RENDERING STUFF #########################################

def downsample(img, factor):
    """
    Downsample an image along both dimensions by some factor
    """

    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0]//factor, factor, img.shape[1]//factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    return img

def fill_coords(img, fn, color):
    """
    Fill pixels of an image with coordinates matching a filter function
    """

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
    p0 = np.array([x0, y0])
    p1 = np.array([x1, y1])
    dir = p1 - p0
    dist = np.linalg.norm(dir)
    dir = dir / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x, y):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = np.array([x, y])
        pq = q - p0

        # Closest point on line
        a = np.dot(pq, dir)
        a = np.clip(a, 0, dist)
        p = p0 + a * dir

        dist_to_line = np.linalg.norm(q - p)
        return dist_to_line <= r

    return fn

def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r * r
    return fn

def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax
    return fn

def point_in_triangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    def fn(x, y):
        v0 = c - a
        v1 = b - a
        v2 = np.array((x, y)) - a

        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v) < 1

    return fn

def highlight_img(img, color=(255, 255, 255), alpha=0.30):
    """
    Add highlighting to an image
    """

    blend_img = img + alpha * (np.array(color, dtype=np.uint8) - img)
    blend_img = blend_img.clip(0, 255).astype(np.uint8)
    img[:, :, :] = blend_img

############################### RENDERING ENDS ###################################

############################### WINDOW STUFF #####################################

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()
        # self.fig.subplots_adjust(bottom=0.2, left=0.1)
        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.001)

    def show_indices(self,agents,height,txts,size,tilesize,color=COLORS['yellow']/255,goals=None):
        """
        Show index of agents 
        """
        # print(txts)
        # print('Show indices')
        # if len(txts)>0:
        #     for txt in txts:
        #         txt.remove()
        goals = np.asarray(goals)
        txt = []
        for i in range(len(agents)):
            # print('Pos of agent {}: {}; Print at : {}'.format(i,agents[i].pos,tilesize*agents[i].pos))
            try:
                if goals[i] is not None:
                    txt.append(self.ax.annotate('{}'.format(i+1),xy=tilesize*goals[i],weight='bold',\
                    xycoords='data',size=size, ha="center", va="center", wrap=True,\
                    transform=self.ax.transAxes,color=color))
            except IndexError:
                pass

            if agents[i].dir == 0: #+x
                xy = tilesize*(agents[i].pos+(1,1))
            elif agents[i].dir == 1: #+y
                xy = tilesize*(agents[i].pos+(0,1))
            elif agents[i].dir == 2: #-x
                xy = tilesize*(agents[i].pos+(0,0))
            else: #-y
                xy = tilesize*(agents[i].pos+(1,0))
            txt.append(self.ax.annotate('{}'.format(i+1),xy=xy,weight='bold',\
                xycoords='data',size=size, ha="center", va="center", wrap=True,transform=self.ax.transAxes,color=color))
        return txt

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def show_scores(self,scores,agents,ne,np):
        """
        Set/update the agent scores below the image
        """
        # for txt in self.fig.texts:
        #     txt.remove()
        num_agents = len(agents)
        num_e = 0
        num_p = 0
        txt = []
        for i in range(num_agents):
            if agents[i].index==1: #evader
                txt.append(self.ax.annotate("Score for player {}: {:.2f}".format(i+1,scores[i]),\
                    xy=(0.8,(num_e+1)/(ne+1)),xycoords='figure fraction',size=10, wrap=True,ha="left", va="center",
                    transform=self.ax.transAxes,bbox=dict(boxstyle="round", fc=COLORS[agents[i].color]/255)))
                num_e += 1
            else: #pursuer
                # plt.text(0,(num_p+1)/(np+1),"Score for player {}: {}".format(i,scores[i]),size=10, wrap=True, ha="left", va="center",
                #     transform=self.ax.transAxes,bbox=dict(boxstyle="round", fc=COLORS[colors[i]]))
                txt.append(self.ax.annotate("Score for player {}: {:.2f}".format(i+1,scores[i]),\
                    xy=(0.2,(num_p+1)/(np+1)),xycoords='figure fraction',size=10, wrap=True, ha="right", va="center",
                    transform=self.ax.transAxes,bbox=dict(boxstyle="round", fc=COLORS[agents[i].color]/255)))
                num_p += 1
        return txt

    def show_message(self,done,message='',size=10,xy=(0.5,0.15),color=COLORS['purple']/255,\
        txts_ind=None):
        """
        Show given message on game window
        """
        # if done and len(txts_ind)>0:
        #     for txt in txts_ind:
        #         txt.remove()
        txt = []
        txt.append(self.ax.annotate(message,xy=xy,xycoords='figure fraction',size=size, ha="center", va="top", wrap=True,
                transform=self.ax.transAxes,bbox=dict(fc=color)))
        return txt

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()

################################ WINDOW ENDS ######################################


class World:

    encode_dim = 6

    normalize_obs = 1

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'agent': 10,
        'objgoal': 11,
        'Pagent': 12,
        'Eagent': 13,
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class SmallWorld:

    encode_dim = 3

    normalize_obs = 1/3

    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'grey': 3
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'agent': 3
    }

    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return True

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return True

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)
        else:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, 0, 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(self, world, index, target_type='ball', color=None):
        if color is None:
            super().__init__(world, 'objgoal', world.IDX_TO_COLOR[index])
        else:
            # super().__init__(world, 'objgoal', world.IDX_TO_COLOR[color])
            super().__init__(world, 'objgoal', color)
        self.target_type = target_type
        self.index = index

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, world, index, color=None):
        if color is None:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[color])
        self.index = index

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world, color='blue'):
        super().__init__(world, 'floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c / 2)
        r.drawPolygon([
            (1, TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS, 1),
            (1, 1)
        ])


    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Wall(WorldObj):
    def __init__(self, world, color='grey'):
        super().__init__(world, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7,vel=1,color=None,global_index=None):
        super(Agent, self).__init__(world, 'agent', color=color)
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.vel = vel
        self.captured = False #flag is true if captured
        self.global_index = global_index

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], self.dir)
        elif self.carrying:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 0)

        else:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 0)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    # @property
    def front_pos(self,width,height,steps=-1):
        """
        Get the position of the cell that is right in front of the agent
        """
        if steps < 0:
            steps = self.vel
        fp = self.pos + int(steps) * self.dir_vec
        if fp[0] < 1:
            fp[0] = 1
        if fp[0] >= width-1:
            fp[0] = width-2
        if fp[1] < 1:
            fp[1] = 1
        if fp[1] >= height-1:
            fp[1] = height-2
        return fp

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self,radius=1):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        # print('In Grid {} {}'.format(width,height))
        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            world,
            obj,
            highlights=[],
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        key = (*highlights, tile_size)
        key = obj.encode(world) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(img, color=COLORS[world.IDX_TO_COLOR[h%len(world.IDX_TO_COLOR)]])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            world,
            tile_size,
            highlight_masks=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    highlights=[] if highlight_masks is None else highlight_masks[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world)

        return array

    def encode_for_agents(self, world, agent_pos, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world, current_agent=np.array_equal(agent_pos, (i, j)))

        return array

    # @staticmethod
    # def decode(array):
    #     """
    #     Decode an array grid encoding back into a grid
    #     """
    
    #     width, height, channels = array.shape
    #     assert channels == 3
    
    #     vis_mask = np.ones(shape=(width, height), dtype=np.bool)
    
    #     grid = Grid(width, height)
    #     for i in range(width):
    #         for j in range(height):
    #             type_idx, color_idx, state = array[i, j]
    #             v = WorldObj.decode(type_idx, color_idx, state)
    #             grid.set(i, j, v)
    #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])
    
    #     return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

class Actions:
    available=['still', 'left', 'right', 'forward']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3
    # # Toggle/activate an object
    # toggle = 4
    # # Done completing task
    # done = 5

####################################### ENV SPECS #######################################

class PursuitEvasionEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions

    def __init__(
            self,
            ep=0,
            height=10,
            width=10,
            max_steps=50,
            partial_obs=False,
            goal_pst = [[8,8]],
            goal_index = [1],
            agent_index = [1,2,2], #1: evader, 2: pursuer
            agent_color = ['red','green','green'],#blue,red
            agent_vel=[1,3,5],
            view_size=[2+3,0+3,0+3],
            zero_sum=True,
            score_factors=[1,1,1e-2,1e-2],#R: reaching goal,C: capture - zero sum, T: long time, D: distance to goal\
            grid_size=None,
            see_through_walls=False,
            seed=None,
            actions_set=Actions,
            objects_set = World,
            ):

        self.goal_pst = goal_pst
        self.goal_index = goal_index
        self.zero_sum = zero_sum

        self.world = World
        self.num_evaders = 0
        self.num_pursuers = 0
        self.ep = ep
        agents = []
        for i in range(len(agent_index)):
            agents.append(Agent(self.world, index=agent_index[i], view_size=view_size[i],vel=agent_vel[i],color=agent_color[i],global_index=i))
            if agent_index[i] == 1:
                self.num_evaders += 1
            else:
                self.num_pursuers += 1

        self.agent_view_size = view_size
        self.agent_index = np.array(agent_index)

        self.agents = agents
        self.num_agents = len(agents)

        # Does the agents have partial or full observation?
        self.partial_obs = partial_obs

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set
        # self.trans_prob = trans_prob #P(s,s',a)

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(4)
        # self.action_space = spaces.Box(low=np.array([0]),high=np.array([3]),dtype=np.int32)

        # print(self.action_space.shape)
        obs_mod = np.array([width-2,height-2]) #goal
        for i in range(self.num_agents):
            obs_mod = np.append(obs_mod,width-2+height-2)

        obs = np.array([])
        for i in range(self.num_agents):
            obs = np.append(obs,[width-2,height-2,4])

        self.objects=objects_set

        # self.observation_space = spaces.MultiDiscrete(obs)

        obs_mod = np.concatenate((obs_mod,obs))
        # print(obs_mod)
        self.observation_space = spaces.MultiDiscrete(obs_mod)

        # self.num_obs = len(obs)
        # low = np.zeros(self.num_obs)
        # high = 3*np.ones(self.num_obs)
        # for i in range(self.num_obs):
        #     if np.remainder(i,3) == 0: #x
        #         low[i] = 1
        #         high[i] = width-2
        #     elif np.remainder(i,3) == 1: #y
        #         low[i] = 1
        #         high[i] = height-2
        # # print(low,high)
        # self.observation_space = spaces.Box(low=low,high=high)

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = len(self.actions.available)

        # Range of possible rewards
        # self.reward_range = (0, 2)
        self.scores = max_steps*score_factors #R:reaching goal, C: capture - zero sum, T: long time, D: distance to goal
        # print('Scores: {}'.format(self.scores))
        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls
        self.agent_color = agent_color
        self.evader_index = np.where(self.agent_index==1)[0]
        self.pursuer_index = np.where(self.agent_index==2)[0]

        self.txts = [] #text boxes from messages/scores
        self.txts_ind = [] #text boxes from agent indices
        # print(agent_index,np.where(agent_index==1))
        # Initialize the RNG
        self.seed(seed=seed)
        self.messages = ''
        self.win_message = ''
        # Initialize the state
        self.reset(ep=ep)
        # time.sleep(5.0)

    def reset(self,ep=1):

        self.ep = ep
        self.window = None
        self.txts = []
        self.txts_ind = []
        self.pursuer_capture = np.zeros((self.num_agents,3)) #flag, iteration and predator index of capture
        self.pursuer_capture[:,2] = 1+self.num_agents #to prevent accidental predator index find
        self.evader_capture = np.zeros((self.num_agents,3)) #flag, iteration and predator index of capture
        self.evader_goal = np.zeros((self.num_agents,2)) #flag, iteration of reaching goal
        self.messages = ''
        self.win_message = ''
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.rewards = np.zeros(self.num_agents)
        # rewards = np.zeros(self.num_agents)
        for i in range(len(self.agents)):
            self.agents[i].terminated = False
            self.agents[i].captured = False

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            # obs = [np.append(self.agents[i].pos,self.agents[i].dir) for i in range(len(actions))]
            obs = np.append(self.agents[0].pos,self.agents[0].dir)
            dist = abs(obs[0]-self.goal_pst[0][0])+abs(obs[1]-self.goal_pst[0][1])
            for i in range(1,self.num_agents):
                obs = np.append(obs,np.append(self.agents[i].pos,self.agents[i].dir))
                dist = np.append(dist,abs(self.agents[0].pos[0]-self.agents[i].pos[0])\
                    +abs(self.agents[0].pos[1]-self.agents[i].pos[1]))

        obs_mod = np.concatenate((np.asarray(self.goal_pst[0]),dist))
        obs_mod = np.concatenate((obs_mod,obs))
        # return obs
        return obs_mod

    def seed(self, seed=None):
        # Seed the random number generator
        if seed is None:
            seed = np.random.randint(2**32)
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _gen_grid(self, width, height):
        # print(width,height)
        
        self.grid = Grid(width, height)
        # print('In gen grid')
        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for i in range(len(self.goal_pst)):
            self.place_obj(ObjectGoal(self.world,self.goal_index[i],color=self.agent_color[i]), top=self.goal_pst[i], size=[1,1])

        # Randomize the player start position and orientation
        for i in range(len(self.agents)):
            self.place_agent(self.agents[i])
            self.agents[i].color = self.agent_color[i]


    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """

        return 1# - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            # print(top[0] + 1, min(top[0] + size[0], self.grid.width - 1))
            if top == (0,0):
                pos = np.array((
                    self._rand_int(top[0] + 1, min(top[0] + size[0], self.grid.width - 1)),
                    self._rand_int(top[1] + 1, min(top[1] + size[1], self.grid.height - 1))
                ))
            else:
                pos = np.array((
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
                ))

            # Don't place the object on top of another object
            # if self.grid.get(*pos) != None:
            #     continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def agent_sees(self, my_pos, other_pos, my_dir, view_size, allow_clash):
        """
        Changed!
        """
        if sum(abs(my_pos-other_pos))==0:
            return allow_clash
        else:
            flag = False
            v = view_size
            w = int(np.floor(view_size/2))

            if my_dir == 0 and other_pos[0]>=my_pos[0] and other_pos[0]<my_pos[0]+v and \
            other_pos[1]>=my_pos[1]-w and other_pos[1]<=my_pos[1]+w: #positive x axis
                flag = True
            if my_dir == 1 and other_pos[1]>=my_pos[1] and other_pos[1]<my_pos[1]+v and \
            other_pos[0]>=my_pos[0]-w and other_pos[0]<=my_pos[0]+w: #positive y axis
                flag = True
            if my_dir == 2 and other_pos[0]<=my_pos[0] and other_pos[0]>my_pos[0]-v and \
            other_pos[1]>=my_pos[1]-w and other_pos[1]<=my_pos[1]+w: #negative x axis
                flag = True
            if my_dir == 3 and other_pos[1]<=my_pos[1] and other_pos[1]>my_pos[1]-v and \
            other_pos[0]>=my_pos[0]-w and other_pos[0]<=my_pos[0]+w: #negative y axis
                flag = True
            return flag


            # if np.nonzero(my_dir_vec[0]): #aligned along x axis
            #     if my_dir_vec[0]>0: #positive x
            #         if other_pos[0]>=my_pos[0] and other_pos[0]<my_pos[0]+v and \
            #         other_pos[1]>=my_pos[1]-w and other_pos[1]<=my_pos[1]+w:
            #             flag = True
            #     else: #negative x
            #         if other_pos[0]<=my_pos[0] and other_pos[0]>my_pos[0]-v and \
            #         other_pos[1]>=my_pos[1]-w and other_pos[1]<=my_pos[1]+w:
            #             flag = True
            # else: #aligned along y axis
            #     if my_dir_vec[1]>0: #positive y
            #         if other_pos[1]>=my_pos[1] and other_pos[1]<my_pos[1]+v and \
            #         other_pos[0]>=my_pos[0]-w and other_pos[0]<=my_pos[0]+w:
            #             flag = True
            #     else: #negative y
            #         if other_pos[1]<=my_pos[1] and other_pos[1]>my_pos[1]-v and \
            #         other_pos[0]>=my_pos[0]-w and other_pos[0]<=my_pos[0]+w:
            #             flag = True
            # return flag


    def reach_goal_action(self,my_pos,goal_pos,my_dir,my_vel):
        action = 3 #forward unless facing wrong direction
        if goal_pos[0]>=my_pos[0]: #goal in Q1 or Q4
            if goal_pos[1]>=my_pos[1]: #goal in Q4
                if my_dir == 3: #facing up
                    action = 2
                elif my_dir == 2: #facing left
                    action = 1
                elif my_dir==0 and my_pos[0]==self.width-2: #facing right and at boundary of grid
                    action = 2
                elif my_dir==1 and my_pos[1]==self.height-2: #facing down and at boundary of grid
                    action = 1
            else: #Q1
                if my_dir == 1: #facing down
                    action = 1
                elif my_dir == 2:#facing left
                    action = 2
                elif my_dir == 3 and my_pos[1]==1: #facing up and at boundary of grid
                    action = 2

        else: #Q2 or Q3         
            if goal_pos[1]>=my_pos[1]: #Q3
                if my_dir == 3: #facing up
                    action = 1
                elif my_dir == 0: #facing right
                    action = 2
                elif my_dir == 2 and my_pos[0]==1: #facing left and at boundary of grid
                    action = 1
            
            else: #Q2
                if my_dir == 1: #facing down
                    action = 2
                elif my_dir == 0: #facing right
                    action = 1
        if action == 3:         
            steps = max(abs(my_pos[0]-goal_pos[0]),abs(my_pos[1]-goal_pos[1]))
            steps = min(my_vel,steps)
        else:
            steps = -1
        return action,steps


    def round_action(self,ev_action):
        ev_action = np.array(ev_action)
        na = ev_action.ndim
        if na == 0:
            na = 1
            ev_action = np.array([ev_action])
        a = np.zeros(na,dtype=int)
        for i in range(na):
            a[i] = round(ev_action[i])
            if a[i] < 0:
                a[i] = 0
            if a[i] > 3:
                a[i] = 3
        return a

    def step(self, ev_action):
        ev_action = np.atleast_1d(ev_action)
        self.step_count += 1
        actions = np.zeros(self.num_agents)
        steps = np.zeros(self.num_agents)
        # order = np.random.permutation(self.num_agents)
        rewards = np.zeros(self.num_agents)
        done = False
        # print('order: {}'.format(order))
        ## Update agent states
        # print('Step: {}'.format(self.step_count))
        # print(ev_action)
        # ev_action = self.round_action(ev_action)
        # print(ev_action)
        for i in range(self.num_agents):
            # print(colored('Agent {}'.format(i),'red'))
            # print('{} Terminated:{}'.format(i,self.agents[i].terminated))
            if self.agents[i].index == 2: #pursuer
                evader_set = [e for e in self.evader_index if not self.agents[e].terminated]
                # print(evader_set)
                closest_evader = evader_set[0] if len(evader_set)>0 else []
                if len(evader_set):
                    dis_closest_evader = sum(abs(self.agents[i].pos-self.agents[closest_evader].pos))
                    for j in evader_set:
                        dis = sum(abs(self.agents[i].pos-self.agents[j].pos))
                        if dis<= dis_closest_evader:
                            dis_closest_evader = dis
                            closest_evader = j
                    actions[i],steps[i] = self.reach_goal_action(self.agents[i].pos,self.agents[closest_evader].pos,self.agents[i].dir,self.agents[i].vel)
                # print('Steps for {}: {}'.format(i,steps[i]))
                fwd_pos = self.agents[i].front_pos(width=self.width,height=self.height,steps=steps[i])
            else:
                if ev_action.size == 1: #1 evader
                    actions[i] = ev_action
                else:
                    actions[i] = ev_action[np.where(self.evader_index==i)[0]]
                fwd_pos = self.agents[i].front_pos(width=self.width,height=self.height)

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[i] == self.actions.still:
                continue

            # print('Action: {}'.format(actions[i]))
            # Get the position in front of the agent
            
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4
                # self.agents[i].dir_vec = DIR_TO_VEC[self.agents[i].dir]
            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4
                # self.agents[i].dir_vec = DIR_TO_VEC[self.agents[i].dir]
            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None) #to update agent state on grid
                    self.agents[i].pos = fwd_pos
                # self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                assert False, "unknown action {} for {}".format(actions[i],i)

        # print('Actions: {}'.format(actions))
        ## Assign agent rewards
        messages = []
        for i in range(self.num_agents):
            # print(colored('Agent {}'.format(i),'blue'))
            # print('{} Terminated:{}'.format(i,self.agents[i].terminated))
            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started:
                continue

            if self.agents[i].index == 1: #evader

                # print(colored('Evader!','red'))
                if sum(abs(self.agents[i].pos-self.goal_pst[i]))<1: #reached goal
                    self.evader_goal[i,0] = 1
                    self.evader_goal[i,1] = self.step_count
                else:
                    rewards[i] -= self.scores[2] #penalise for long time maneuvers
                    rewards[i] -= self.scores[3]*sum(abs(self.agents[i].pos-self.goal_pst[i])) #penalise by distance to goal
                    for j in self.pursuer_index: #agent_sees(self, my_pos, other_pos, my_dir, view_size, allow_clash)
                        if self.agent_sees(self.agents[i].pos,self.agents[j].pos,self.agents[i].dir,self.agents[i].view_size,allow_clash=False): #evader i sees pursuer j
                            self.pursuer_capture[j,0] = 1
                            self.pursuer_capture[j,1] = self.step_count
                            self.pursuer_capture[j,2] = i
                            self.agents[j].terminated = True
                            self.agents[j].captured = True
                            self.agents[j].color = 'yellow'
                            if not self.agents[j].terminated:
                                rewards[j] -= self.scores[1] #captured!
                            messages.append('Your member {} caught my member {}!'.format(i+1,j+1))

                if self.evader_goal[i,0]: #evader i reached its goal
                    preys_i = np.where(self.pursuer_capture[:,2]==i)[0] #preys for evader i
                    self.agents[i].terminated = True
                    if len(preys_i)>0: #evader i captured at least one pursuer
                        for j in preys_i: #all pursuers captured by evader i
                            # if self.pursuer_capture[j,1] == self.step_count: #get reward only at capture
                            rewards[i] += self.scores[0]+self.scores[1]
                            messages.append('Your member {} caught my member {} and reached its goal!'.format(i+1,j+1))
                            # self.render(ep=self.ep,message='Your member {} caught my member {} and reached its goal!'.format(i,j),size=10,xy=(0.5,0.1))
                    else: #evader only reached its goal
                        rewards[i] += self.scores[0] 
                        messages.append('Your member {} reached its goal!'.format(i+1))
                        # self.render(ep=self.ep,message='Your member {} reached its goal!'.format(i),size=10,xy=(0.5,0.1))
                        
            else: #pursuer
                # print(colored('Pursuer','green'))
                for j in self.evader_index:
                    if self.agent_sees(self.agents[i].pos,self.agents[j].pos,self.agents[i].dir,self.agents[i].view_size,allow_clash=True): #pursuer i sees evader j
                        self.evader_capture[j,0] = 1
                        self.evader_capture[j,1] = self.step_count
                        self.evader_capture[j,2] = i
                preys_i = np.where(self.evader_capture[:,2]==i)[0] #preys for pursuer i
                if len(preys_i)>0: #pursuer i captured at least one evader
                    for j in preys_i: #all pursuers captured by evader i
                        # if self.evader_capture[j,1] == self.step_count: #get reward only at capture
                        rewards[i] += self.scores[1]
                        if self.agents[j].terminated:
                            rewards[j] -= self.scores[1]
                        self.agents[j].terminated = True
                        self.agents[j].captured = True
                        self.agents[j].color = 'yellow'
                        messages.append('My member {} caught your member {}!'.format(i+1,j+1))
                        # self.render(ep=self.ep,message='My member {} caught your member {}!'.format(i,j),size=10,color='green',xy=(0.5,0.1))
        # print('Rewards:{}'.format(rewards))
        evaders_terminated = 0
        evaders_captured = 0 #subset of terminated evaders
        for i in self.evader_index:
            if self.agents[i].terminated:
                evaders_terminated += 1
                if self.agents[i].captured:
                    evaders_captured += 1
        pursuers_terminated = 0 #pursuers are terminated only when captured
        for i in self.pursuer_index:
            if self.agents[i].terminated:
                pursuers_terminated += 1

        self.rewards += rewards
        
        if evaders_terminated == self.num_evaders or self.step_count >= self.max_steps: #all evaders terminated
            done = True

        self.messages = messages

        e_win = False
        win_message = []
        if evaders_captured  == self.num_evaders: #all evaders captured
            self.win_message = 'I win with score {:.2f}! You lost all of your team with score: {:.2f}'.\
                format(sum(self.rewards[self.pursuer_index]),sum(self.rewards[self.evader_index]))

        elif done: #either time up or all e at goal or all e terminated
            if evaders_terminated == self.num_evaders: #some e captured, others at goal
                e_win = True
                self.win_message = 'You win with score {:.2f}! I lost {}/{} members with score: {:.2f}'\
                    .format(sum(self.rewards[self.evader_index]),int(pursuers_terminated),\
                        int(self.num_pursuers),sum(self.rewards[self.pursuer_index]))
            else:
                self.win_message = 'Time\'s up! You lost with score: {:.2f}'.\
                    format(sum(self.rewards[self.evader_index]))

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            # obs = [np.append(self.agents[i].pos,self.agents[i].dir) for i in range(len(actions))]
            obs = np.append(self.agents[0].pos,self.agents[0].dir)
            dist = abs(obs[0]-self.goal_pst[0][0])+abs(obs[1]-self.goal_pst[0][1])
            for i in range(1,len(actions)):
                obs = np.append(obs,np.append(self.agents[i].pos,self.agents[i].dir))
                dist = np.append(dist,abs(self.agents[0].pos[0]-self.agents[i].pos[0])\
                    +abs(self.agents[0].pos[1]-self.agents[i].pos[1]))
            
        obs_mod = np.concatenate((np.asarray(self.goal_pst[0]),dist))
        obs_mod = np.concatenate((obs_mod,obs))

        # if done:
        #     time.sleep(0.5)
            # self.render(close=True)
        # return obs, rewards[self.evader_index], done, e_win
        return obs_mod, rewards[self.evader_index], done, e_win

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []
        # print(self.objects)
        # print(self.grid)
        for a in self.agents:

            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(agent_pos=(a.view_size // 2, a.view_size - 1))
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grids, vis_masks = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        obs = [grid.encode_for_agents(self.objects, [grid.width // 2, grid.height - 1], vis_mask) for grid, vis_mask in zip(grids, vis_masks)]

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            self.objects,
            tile_size,
            highlight_mask=vis_mask
        )

        return img

    # def throw_message(self,message=None,tile_size=TILE_PIXELS,size=10,xy=(0.5,0.1),color='purple'):
    #     self.window = Window('gym_multigrid')
    #     self.window.show()
    #     # Render the whole grid
    #     img = self.grid.render(
    #         self.objects,
    #         tile_size,
    #         highlight_masks=None
    #     )
    #     self.window.show_img(img)
    #     txts = self.window.show_message(message,size=size,xy=xy,color=color)
    #     self.txts.append(txts)

    def render(self, mode='human', close=False, highlight=True,tile_size=TILE_PIXELS,ep=None,scores=None,\
        size=30,xy=(0.5,0.5),color='red',done=False):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                for txts in self.txts:
                    txts.remove()
                    # for i,a in enumerate(txts):
                    #     a.remove()
                self.window.close()
            return

        if mode == 'human' and not self.window:
            self.window = Window('gym_multigrid')
            self.window.show(block=False)

        if highlight:

            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {(i, j): [] for i in range(self.width) for j in range(self.height)}

            for i, a in enumerate(self.agents):

                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            self.objects,
            tile_size,
            highlight_masks=highlight_masks if highlight else None
        )

        if mode == 'human':
            message = self.messages
            # print('SELF TXTS:\n {}, len: {}'.format(self.txts,len(self.txts)))
            if len(self.txts)>0:
                for txt in self.txts[:]:
                    txt.remove()
            self.txts = []
            txts = self.window.show_indices(agents=self.agents,height=self.grid.height,txts=self.txts_ind,size=10,\
            tilesize=TILE_PIXELS,goals=self.goal_pst) #display agent and goal indices
            self.txts.extend(txts) 
            txts = self.window.show_scores(scores=self.rewards,agents=self.agents,ne=self.num_evaders,np=self.num_pursuers)
            self.txts.extend(txts) #display agent scores
            # txts_ind = self.window.show_indices(agents=self.agents,height=self.grid.height,txts=self.txts_ind,size=10,tilesize=TILE_PIXELS)
            # self.txts_ind = txts_ind
            # if message:
            for i in range(len(message)):
                txts = self.window.show_message(message=message[i],size=10,xy=(0.5,0.1),\
                    color='yellow',txts_ind=self.txts_ind,done=done)
                self.txts.extend(txts)
            if self.win_message:
                txts = self.window.show_message(message=self.win_message,size=13,xy=(0.5,0.1),\
                    color='grey',txts_ind=self.txts_ind,done=done)
                self.txts.extend(txts)
            self.window.ax.set_title('Game: {}'.format(ep),size=20, ha="center", va="center",
                bbox=dict(boxstyle="round", fc=COLORS['grey']/255))
            self.window.show_img(img)

        return img
