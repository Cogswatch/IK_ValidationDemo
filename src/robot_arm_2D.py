#!/usr/bin/env python3

# Get the windowing packages
from itertools import accumulate
import quopri
from matplotlib.pyplot import axis
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGroupBox, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize

from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
import timeit

from random import random

import numpy as np
from numpy import sin, cos, pi


# A helper class that implements a slider with given start and end value; displays values
class SliderDisplay(QWidget):
    gui = None

    def __init__(self, name, low, high, initial_value, ticks=500):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        :param ticks: Resolution of slider - all sliders are integer/fixed number of ticks
        """
        # Save input values
        self.name = name
        self.low = low
        self.range = high - low
        self.ticks = ticks

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        # call back - calls change_value when slider changed/moved
        self.slider.valueChanged.connect(self.change_value)

        # For displaying the numeric value
        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return (self.slider.value() / self.ticks) * self.range + self.low

    # Called when the value changes - resets display text
    def change_value(self):
        if SliderDisplay.gui:
            SliderDisplay.gui.repaint()
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))

    # Use this to change the value (does clamping)
    def set_value(self, value_f):
        """Set the value of the slider
        @param value_f: value between low and high - clamps if not"""
        value_tick = self.ticks * (value_f - self.low) / self.range
        value_tick = min(max(0, value_tick), self.ticks)
        self.slider.setValue(int(value_tick))
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))


# The main class for handling the robot drawing and geometry
class DrawRobot(QWidget):
    def __init__(self, in_gui):
        super().__init__()

        # In order to get to the slider values
        self.gui = in_gui

        # Title of the window
        self.title = "Robot arm"
        # output text displayed in window
        self.text = "Not reaching"

        # Window size
        self.top = 15
        self.left = 15
        self.width = 500
        self.height = 500

        # For doing dictionaries
        self.components = ['1', '2', '3', '4', '5', '6']
        # Set geometry
        self.init_ui()

    def init_ui(self):
        self.text = "Not reaching"
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # For making sure the window shows up the right size
    def minimumSizeHint(self):
        return QSize(self.width, self.height)

    # For making sure the window shows up the right size
    def sizeHint(self):
        return QSize(self.width, self.height)

    # What to draw - called whenever window needs to be drawn
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw_text(event, qp)
        self.draw_target(qp)
        self.draw_arm(qp)
        qp.end()

    # Put some text in the bottom left
    def draw_text(self, event, qp):
        qp.setPen(QColor(168, 34, 3))
        qp.setFont(QFont('Decorative', 10))
        qp.drawText(event.rect(), Qt.AlignBottom, self.text)

    # Map from [0,1]x[0,1] to the width and height of the window
    def x_map(self, x):
        return int(x * self.width)

    # Map from [0,1]x[0,1] to the width and height of the window - need to flip y
    def y_map(self, y):
        return self.height - int(y * self.height) - 1

    # Draw a + where the target is and another where the end effector is
    def draw_target(self, qp):
        pen = QPen(Qt.darkGreen, 2, Qt.SolidLine)
        qp.setPen(pen)
        x_i = self.x_map(self.gui.reach_x.value())
        y_i = self.y_map(self.gui.reach_y.value())
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

        pt = self.arm_end_pt()
        pen.setColor(Qt.darkRed)
        qp.setPen(pen)

        x_i = self.x_map(pt[0])
        y_i = self.y_map(pt[1])
        qp.drawLine(x_i-5, y_i, x_i+5, y_i)
        qp.drawLine(x_i, y_i-5, x_i, y_i+5)

    # Make a rectangle with the center at the middle of the left hand edge
    # Width is 1/4 length
    # returns four corners with points as row vectors
    @staticmethod
    def make_rect(in_len):
        """Draw a rectangle of the given length; width is 1/4 of length
        @param: in_len desired length
        @return: a 1x4 array of x,y values representing the four corners of the rectangle"""
        x_l = 0
        x_r = in_len
        h = in_len/4
        y_b = -h/2
        y_t = y_b + h
        return [[x_l, y_b, 1], [x_r, y_b, 1], [x_r, y_t, 1], [x_l, y_t, 1]]

    # Apply the matrix m to the points in rect
    @staticmethod
    def transform_rect(rect, m):
        """Apply the 3x3 transformation matrix to the rectangle
        @param: rect: Rectangle from make_rect
        @param: m - 3x3 matrix
        @return: a 1x4 array of x,y values of the transformed rectangle"""
        rect_t = []
        for p in rect:
            p_new = m @ np.transpose(p)
            rect_t.append(np.transpose(p_new))
        return rect_t

    # Create a rotation matrix
    @staticmethod
    def rotation_matrix(theta):
        """Create a 3x3 rotation matrix that rotates in the x,y plane
        @param: theta - amount to rotate by in radians
        @return: 3x3 matrix, 2D rotation plus identity """
        m_rot = np.identity(3)
        m_rot[0][0] = cos(theta)
        m_rot[0][1] = -sin(theta)
        m_rot[1][0] = sin(theta)
        m_rot[1][1] = cos(theta)
        return m_rot

    # Create a translation matrix
    @staticmethod
    def translation_matrix(dx, dy):
        """Create a 3x3 translation matrix that moves by dx, dy
        @param: dx - translate by that much in x
        @param: dy - translate by that much in y
        @return: 3x3 matrix """
        m_trans = np.identity(3)
        m_trans[0, 2] = dx
        m_trans[1, 2] = dy
        return m_trans

    # Draw the given box
    def draw_rect(self, rect, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        for i in range(0, len(rect)):
            i_next = (i+1) % len(rect)
            x_i = self.x_map(rect[i][0])
            y_i = self.y_map(rect[i][1])
            x_i_next = self.x_map(rect[i_next][0])
            y_i_next = self.y_map(rect[i_next][1])
            qp.drawLine(x_i, y_i, x_i_next, y_i_next)

    # Return the matrices that move each of the components. Do this as a dictionary, just to be clean
    def get_matrices(self):
        # The values used to build the matrices
        L1 = self.gui.L1.value()
        L2 = self.gui.L2.value()
        L3 = self.gui.L3.value()
        L4 = self.gui.L4.value()
        L5 = self.gui.L5.value()
        L6 = self.gui.L6.value()

        R1 = self.gui.R1.value()
        R2 = self.gui.R2.value()
        R3 = self.gui.R3.value()
        R4 = self.gui.R4.value()
        R5 = self.gui.R5.value()
        R6 = self.gui.R6.value()

        mat_ret = dict()

        # begin homework 1 : Problem 2
        # Each of these should be of the form: Translation * rotation

        mat_ret['L1'] = self.translation_matrix(L1,0)
        mat_ret['R1'] = self.rotation_matrix(R1)
        mat_ret['T1'] = (R1)

        mat_ret['L2'] = self.translation_matrix(L2,0)
        mat_ret['R2'] = self.rotation_matrix(R2)
        mat_ret['T2'] = (R2)

        mat_ret['L3'] = self.translation_matrix(L3,0)
        mat_ret['R3'] = self.rotation_matrix(R3)
        mat_ret['T3'] = (R3)

        mat_ret['L4'] = self.translation_matrix(L4,0)
        mat_ret['R4'] = self.rotation_matrix(R4)
        mat_ret['T4'] = (R4)

        mat_ret['L5'] = self.translation_matrix(L5,0)
        mat_ret['R5'] = self.rotation_matrix(R5)
        mat_ret['T5'] = (R5)

        mat_ret['L6'] = self.translation_matrix(L6,0)
        mat_ret['R6'] = self.rotation_matrix(R6)
        mat_ret['T6'] = (R6)

        # end homework 1 : Problem 2
        return mat_ret

    def draw_arm(self, qp):
        """Draw the arm as boxes
        :param: qp - the painter window
        """
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)

        # Create a rectangle for each component then move it to the correct place then draw it
        rects = dict()
        rects['1'] = self.make_rect(self.gui.L1.value())
        rects['2'] = self.make_rect(self.gui.L2.value())
        rects['3'] = self.make_rect(self.gui.L3.value())
        rects['4'] = self.make_rect(self.gui.L4.value())
        rects['5'] = self.make_rect(self.gui.L5.value())
        rects['6'] = self.make_rect(self.gui.L6.value())

        # begin homework 1 : Problem 2
        # Transform and draw each component using the matrices in self.get_matrices()
        # Example call:
        #   rect_transform = self.transform_rect(rects['base'], mat)
        #   self.draw_rect(rect_transform, qp)
            #   getting the translation matrix for upper arm: matrices['L1' + '_T']

        transformations = self.get_matrices()

        apply_transformation        = lambda rect, trans: self.transform_rect(rects[rect], transformations[trans])
        apply_transformation_both   = lambda rect, trans: self.transform_rect(self.transform_rect(rects[rect], transformations['L' + trans]), transformations['R' + trans])

        for i in range(len(list(self.components))):
            # Print Validation
            # s = 'R' + list(self.components)[i]
            # Render Step
            rects[list(self.components)[i]] = apply_transformation(list(self.components)[i], 'R' + list(self.components)[i])

            for j in reversed(range(i)):
                # Print Validation
                # s = s + 'L' + list(self.components)[j] + 'R' + list(self.components)[j]
                # Render Step
                rects[list(self.components)[i]] = apply_transformation_both(list(self.components)[i], list(self.components)[j])
            
            # Print Validation
            # print(s)
            # Render Step
            self.draw_rect(rects[list(self.components)[i]], qp)

    def arm_end_pt(self):
        """ Return the end point of the arm"""
        matrices = self.get_matrices()
        mat_accum = np.identity(3)
        # begin homework 1 : Problem 3

        for i in self.components:
            # print(i)
            mat_accum = mat_accum @ matrices['R' + i]
            mat_accum = mat_accum @ matrices['L' + i]

        # end homework 1 : Problem 3
        pt_end = mat_accum[0:2, 2]
        return pt_end


class RobotArmGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('ROB 514 2D robot arm')

        # Control buttons for the interface
        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Different do reach commands
        reach_gradient_button = QPushButton('Reach gradient')
        reach_gradient_button.clicked.connect(self.reach_gradient)

        reach_jacobian_button = QPushButton('Reach Jacobian')
        reach_jacobian_button.clicked.connect(self.reach_jacobian)

        reaches = QGroupBox('Reaches')
        reaches_layout = QVBoxLayout()
        reaches_layout.addWidget(reach_gradient_button)
        reaches_layout.addWidget(reach_jacobian_button)
        reaches.setLayout(reaches_layout)

        # The parameters of the robot arm we're simulating
        parameters = QGroupBox('Arm parameters')
        parameter_layout = QVBoxLayout()
        self.R1 = SliderDisplay('R1', -np.pi/2, np.pi/2, 0)
        self.R2 = SliderDisplay('R2', -np.pi/2, np.pi/2, 0)
        self.R3 = SliderDisplay('R3', -np.pi/2, np.pi/2, 0)
        self.R4 = SliderDisplay('R4', -np.pi/2, np.pi/2, 0)
        self.R5 = SliderDisplay('R5', -np.pi/2, np.pi/2, 0)
        self.R6 = SliderDisplay('R6', -np.pi/2, np.pi/2, 0)
        self.L1 = SliderDisplay('L1', 0.1, 0.2, 0.1)
        self.L2 = SliderDisplay('L2', 0.1, 0.2, 0.1)
        self.L3 = SliderDisplay('L3', 0.1, 0.2, 0.1)
        self.L4 = SliderDisplay('L4', 0.1, 0.2, 0.1)
        self.L5 = SliderDisplay('L5', 0.1, 0.2, 0.1)
        self.L6 = SliderDisplay('L6', 0.1, 0.2, 0.1)
        self.theta_slds = []
        self.theta_slds.append(self.R1)
        self.theta_slds.append(self.R2)
        self.theta_slds.append(self.R3)

        parameter_layout.addWidget(self.R1)
        parameter_layout.addWidget(self.R2)
        parameter_layout.addWidget(self.R3)
        parameter_layout.addWidget(self.R4)
        parameter_layout.addWidget(self.R5)
        parameter_layout.addWidget(self.R6)

        parameter_layout.addWidget(self.L1)
        parameter_layout.addWidget(self.L2)
        parameter_layout.addWidget(self.L3)
        parameter_layout.addWidget(self.L4)
        parameter_layout.addWidget(self.L5)
        parameter_layout.addWidget(self.L6)

        parameters.setLayout(parameter_layout)

        # The point to reach to
        reach_point = QGroupBox('Reach point')
        reach_point_layout = QVBoxLayout()
        self.reach_x = SliderDisplay('x', 0, 1, 0.5)
        self.reach_y = SliderDisplay('y', 0, 1, 0.5)
        random_button = QPushButton('Random')
        random_button.clicked.connect(self.random_reach)
        reach_point_layout.addWidget(self.reach_x)
        reach_point_layout.addWidget(self.reach_y)
        reach_point_layout.addWidget(random_button)
        reach_point.setLayout(reach_point_layout)

        # The display for the graph
        self.robot_arm = DrawRobot(self)

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)
        left_side_layout = QVBoxLayout()
        right_side_layout = QVBoxLayout()

        left_side_layout.addWidget(reaches)
        left_side_layout.addWidget(reach_point)
        left_side_layout.addStretch()
        left_side_layout.addWidget(parameters)

        right_side_layout.addWidget(self.robot_arm)
        right_side_layout.addWidget(quit_button)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderDisplay.gui = self

    # generate a random reach point
    def random_reach(self):
        self.reach_x.set_value(random())
        self.reach_y.set_value(random())
        self.robot_arm.repaint()

    def reach_gradient(self):
        """Align the robot end point (palm) to the target point using gradient descent"""

        # Use the text field to say what happened
        self.robot_arm.text = "Not improved"

        # begin homework 2 : Problem 1
        # Keep trying smaller increments while nothing improves
            # calculate the current distance
            # Try each angle in turn
                # Gradient
        # end homework 2 : Problem 1
        self.robot_arm.repaint()

    def reach_jacobian(self):
        """ Use the Jacobian to calculate the desired angle change"""

        # Get matricies
        transformations = self.robot_arm.get_matrices()

        # Helper
        compName = lambda x: list(self.robot_arm.components)[x]

        # Get initial pose
        poseInitial = np.zeros(6)
        for i in range(len(self.robot_arm.components)):
            poseInitial[i] = transformations['T' + compName(i)]

        # Task Target
        targetPosition = np.zeros(3)
        targetPosition[0] = gui.reach_x.value()
        targetPosition[1] = gui.reach_y.value()
        targetPosition[2] = 1


        # Task Initial
        endEffectorPos = np.zeros(3)
        endEffectorPos[0] = self.robot_arm.arm_end_pt()[0]
        endEffectorPos[1] = self.robot_arm.arm_end_pt()[1]
        endEffectorPos[2] = 1

        # Debug
        # print(endEffectorPos)
        # print(targetPosition)

        start = timeit.default_timer()

        # Get Task Space positions of each joint
        Jt = np.zeros([6,3])

        for i in range(len(list(self.robot_arm.components))):
            accumulate = np.identity(3)
            accumulate = accumulate @ transformations['R' + compName(i)]

            for j in reversed(range(i)):
                accumulate = accumulate @ transformations['R' + compName(j)]
                accumulate = accumulate @ transformations['L' + compName(j)]

            Jt[i] = accumulate[0:3, 2]

        JtP = (endEffectorPos - Jt)
        Jt = np.cross(Jt, JtP)


        # V
        V = targetPosition - endEffectorPos

        dO = Jt @ V

        h = 0.1

        dOH = dO * h

        stop = timeit.default_timer()

        print("=== NEW FRAME ===")

        print('Time: ', stop - start, " < ", 1/30) 

        print("Output 6DOF vector:")
        print(dO)

        print("Input 3DOF vector:")
        print(V)

        self.R1.set_value(self.R1.value() + dOH[0])
        self.R2.set_value(self.R1.value() + dOH[1])
        self.R3.set_value(self.R1.value() + dOH[2])
        self.R4.set_value(self.R1.value() + dOH[3])
        self.R5.set_value(self.R1.value() + dOH[4])
        self.R6.set_value(self.R1.value() + dOH[5])
            
        self.robot_arm.repaint()

    def draw(self, unused_data):
        self.robot_arm.draw()


if __name__ == '__main__':
    app = QApplication([])

    gui = RobotArmGUI()

    gui.show()

    app.exec_()
