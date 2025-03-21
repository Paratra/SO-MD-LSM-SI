import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pdb import set_trace as st

class AngleSpacingCalculator:
    def __init__(self, image, dx):
        self.image = image
        self.image_size = image.shape[0]
        self.dx = dx
        self.xc, self.yc = int(image.shape[0]/2), int(image.shape[1]/2)
        self.fig, self.ax = plt.subplots()
        self.rect = Rectangle((0,0), 0, 0, fill=False, edgecolor='red', linewidth=5)
        self.ax.add_patch(self.rect)
        self.ax.imshow(image, interpolation='nearest')

        self.start_x = None
        self.start_y = None
        self.brightest_pixel = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.start_x, self.start_y = event.xdata, event.ydata
        self.rect.set_width(0)
        self.rect.set_height(0)
        self.rect.set_xy((self.start_x, self.start_y))

    def on_drag(self, event):
        if self.start_x is None or self.start_y is None or event.inaxes != self.ax:
            return
        width = event.xdata - self.start_x
        height = event.ydata - self.start_y
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.fig.canvas.draw()

    def on_release(self, event):
        if event.inaxes != self.ax:
            return
        self.process_selection(event.xdata, event.ydata)

    def process_selection(self, end_x, end_y):
        x1, y1, x2, y2 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y)), int(max(self.start_x, end_x)), int(max(self.start_y, end_y))
        cropped_image = self.image[y1:y2, x1:x2]
        if cropped_image.size > 0:
            brightest_y, brightest_x = np.unravel_index(np.argmax(cropped_image), cropped_image.shape)
            self.brightest_pixel = (brightest_x + x1, brightest_y + y1)
            print("Brightest pixel coordinates in selected area:", self.brightest_pixel)

    def on_key_press(self, event):
        if event.key == 'enter':
            if self.start_x is not None and self.start_y is not None:
                self.process_selection(self.start_x, self.start_y)  # Process with the current start position
                print('confirmed!')
                print("The x and y are set to", self.brightest_pixel)
                self.xb = self.brightest_pixel[0]
                self.yb = self.brightest_pixel[1]


            else: 
                print('No area is seleted yet.')

    def show(self):
        plt.show()

    def get_brightest_pixel(self):
        return self.brightest_pixel
    
    def get_ang_spacing_given_brightest_pixel(self, bp):
        self.brightest_pixel = bp
        self.xb = self.brightest_pixel[0]
        self.yb = self.brightest_pixel[1]
        return self.get_ang_spacing()
            
    def get_ang_spacing(self):
        ang = np.arctan2((self.xb-self.xc),(-self.yb+self.yc))
        spacing = 1/(np.sqrt((self.xb-self.xc)**2+(self.yb-self.yc)**2) * 1/self.image_size/self.dx)

        return -ang, spacing

    # def get_spacing(self):
    #     spacing = 1/(np.sqrt((self.xb-self.xc)**2+(self.yb-self.yc)**2) * 1/self.image_size/self.dx)
    #     return spacing

# Example usage:
# numpy_image = [Your NumPy image data here]
# selector = AngleSpacingCalculator(numpy_image)
# selector.show()
# brightest_pixel = selector.get_brightest_pixel()
# print("The brightest pixel is at:", brightest_pixel)
