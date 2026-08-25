# -*- coding: utf-8 -*-
"""This module contains various functions which correct the magnet x and y positions before the robot places them on the Hector plate. 
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sympy import symbols, Eq, solve
import math


pd.options.mode.chained_assignment = (
    None  # disabled warning about writes making it back to the original frame
)


def pick_up_arm_rotation_correction(
    robot_centre_x, robot_centre_y, rot_platePlacing, sign="positive"
):
    """
    Correct for the errors in magnet position introduced by the different centres of rotation of the robot cylinder and the pickup arm.

    First written by Stefania Barsanti, May 2022. Edited by Sam Vaughan, May/June 2022

    Args:
        robot_centre_x (float): Magnet x coordinate
        robot_centre_y (float): Magnet y coordinate
        rot_platePlacing (float): Theta coordinate of robot arm when placing each magnet. Should be from the rot_platePlacing column in the robot file
        sign (str): Must be "positve" or "negative". Apply a factor of +/- 1 to swap the direction of this correction.

    Returns:
        tuple: The new magnet x coordinate and y coordinate
    """
    if sign == "positive":
        factor = 1
    elif sign == "negative":
        factor = -1
    else:
        raise NameError("Sign must be positive or negative!")
    # d = (
    #     factor * 24 * 0.001
    # )  # distance from center of pick up arm to center of rotation in [um]
    #d = (
    #    factor * 1 * 0.001
    #)  # distance from center of pick up arm to center of rotation in [um]. Changed on September 28th 2023 to be 1 micron after Robot Realignment
    #d = (
    #    factor * 7 * 0.001
    #)  # distance from center of pick up arm to center of rotation in [um]. Changed on April 2024 to be 7 micron after Robot Realignment
    #d = (
    #    factor * 3.5 * 0.001
    #)  # distance from center of pick up arm to center of rotation in [um]. Changed on Oct 2024 to be 3.5 micron after Robot Realignment
    #d = (
    #    factor * 4.2 * 0.001
    #)  # distance from center of pick up arm to center of rotation in [um]. Changed on Jul 2025 to be 4.2 micron after Robot Realignment
    d = (
        factor * 2.9 * 0.001
    )  # distance from center of pick up arm to center of rotation in [um]. Changed on Jun 2026 to be 2.9 micron after Robot Realignment

    # ang0 = 20  # when the pickup arm is rotated to give the maximum shift above, the rotation axis is actually ang0 deg from the +x axis. The rotation direction is clockwise
    #ang0 = 90  # changed on April 2024
    ang0 = 180  # changed on Oct 2024
    theta = np.radians(rot_platePlacing - ang0)

    robot_centre_x_new = robot_centre_x - d * np.cos(theta)
    robot_centre_y_new = robot_centre_y + d * np.sin(theta)

    return robot_centre_x_new, robot_centre_y_new


def apply_offsets_to_magnets(
    df, offset, robot_centre, apply_telecentricity_correction=True, verbose=True
):
    """
    Apply the radial and telecentricity offsets to the circular magnets, then calculate the corresponding positions of the rectangular magnets.

    Args:
        df (dataframe): A dataframe made from reading in the robot file
        offset (float): A given offset distance in mm to move each magnet in the radial direction. Positive is outwards.
        robot_centre (list): A two element list containing the plate centre in robot coordinates.
        apply_telecentricity_correction (bool, optional): If True, apply the telescentricity correction term
        verbose (bool, optional): If True, print information to the screen.

    Returns:
        dataframe: The updated dataframe containing positions with the offsets applied
    """

    if verbose:
        print(
            f"\tApplying an overall radial offset of {offset}mm to all circular magnets"
        )

    # Loop through the hexabundles
    all_hexabundles = df.loc[df["Magnet"] == "circular_magnet", "Hexabundle"].values
    for hexabundle in all_hexabundles:
        circular_magnet = df.loc[
            (df["Hexabundle"] == hexabundle) & (df["Magnet"] == "circular_magnet")
        ]
        rectangular_magnet = df.loc[
            (df["Hexabundle"] == hexabundle) & (df["Magnet"] == "rectangular_magnet")
        ]

        new_x, new_y = calculate_radial_offset(
            circular_magnet,
            offset,
            robot_centre,
            apply_telecentricity_correction=apply_telecentricity_correction,
            verbose=verbose,
        )

        if verbose:
            print(f"\tHexabundle {hexabundle}: \n\tCircular magnet:")
            print(
                f"\t\toriginal centre is ({circular_magnet['Center_x'].values[0]:.3f}, {circular_magnet['Center_y'].values[0]:.3f})"
            )
            print(
                f"\t\tnew centre is ({new_x + robot_centre[0]:.3f}, {new_y + robot_centre[1]:.3f})"
            )

        # Update the new centres of the circular magnets
        df.at[circular_magnet.index[0], "Center_x"] = new_x + robot_centre[0]
        df.at[circular_magnet.index[0], "Center_y"] = new_y + robot_centre[1]

        # Now find the centres of the rectangular magnets
        angle_for_rectangular_magnet = np.radians(
            270
            - rectangular_magnet["rot_holdingPosition"]
            - rectangular_magnet["rot_platePlacing"]
        ).values[0]
        [x_rect, y_rect] = calculate_rectangular_magnet_centre_coordinates(
            new_x, new_y, angle_for_rectangular_magnet
        )

        if verbose:
            print("\tRectangular magnet:")
            print(
                f"\t\toriginal centre is ({rectangular_magnet['Center_x'].values[0]:.3f}, {rectangular_magnet['Center_y'].values[0]:.3f})"
            )
            print(
                f"\t\tnew centre is ({x_rect + robot_centre[0]:.3f}, {y_rect + robot_centre[1]:.3f})"
            )

        # Update the new centres of the rectangular magnets
        df.at[rectangular_magnet.index[0], "Center_x"] = x_rect + robot_centre[0]
        df.at[rectangular_magnet.index[0], "Center_y"] = y_rect + robot_centre[1]

    return df


def calculate_telecentricity_correction(magnet, robot_centre, verbose=True):
    """
    Find the required offset in the position of the circular magnets based on their telecentricity annulus
    """
    if verbose:
        print("\tApplying the telecentricity offset")
    robot_centre_x, robot_centre_y = robot_centre

    centre_x = magnet.Center_x.values
    centre_y = magnet.Center_y.values
    telecentricity = magnet.Label.values

    if telecentricity == "Blu":
        radial_d = 20.7 * 0.001  # mm
    elif telecentricity == "Gre":
        radial_d = 42.0 * 0.001
    elif telecentricity == "Yel":
        radial_d = 61.2 * 0.001
    elif telecentricity == "Mag":
        radial_d = 79.9 * 0.001

    if ((robot_centre_x - centre_x) == 0.0) & ((robot_centre_y - centre_y) == 0.0):
        return 0.0, 0.0

    norm = np.sqrt((robot_centre_x - centre_x) ** 2 + (robot_centre_y - centre_y) ** 2)
    telentricity_offset_x = radial_d * (robot_centre_x - centre_x) / norm
    telentricity_offset_y = radial_d * (robot_centre_y - centre_y) / norm

    return telentricity_offset_x[0], telentricity_offset_y[0]


def calculate_radial_offset(
    circular_magnet,
    radial_offset,
    robot_centre,
    apply_telecentricity_correction=True,
    verbose=True,
):
    """
    Given an x and y coordinate, calculate the total radial offsets to apply based on a temperature change and a fixed telecentricity correction
    """
    x = (circular_magnet["Center_x"] - robot_centre[0]).values[0]
    y = (circular_magnet["Center_y"] - robot_centre[1]).values[0]

    # If x and y are already at the plate centre, return them as they are since we don't know the direction to move them in (their radial offset direction isn't known)
    # Change this to be less than 1e-100 since values like 1e-162 were failing this check
    if (np.abs(x) <= 1e-150) & (np.abs(y) <= 1e-150):
        return 0, 0

    if apply_telecentricity_correction:
        (
            telecentricity_x_corr,
            telecentricity_y_corr,
        ) = calculate_telecentricity_correction(
            circular_magnet, robot_centre, verbose=verbose
        )
    else:
        telecentricity_x_corr = 0
        telecentricity_y_corr = 0

    # Make a unit vector which points towards the origin
    # The origin here is (0, 0) since we've already removed the plate centre offset
    unit_vector = 1.0 / (np.sqrt(x**2 + y**2)) * np.array([x, y]) - np.array([0, 0])
    offset_vector = unit_vector * radial_offset

    delta_x = offset_vector[0]
    delta_y = offset_vector[1]

    new_x = x + delta_x + telecentricity_x_corr
    new_y = y + delta_y + telecentricity_y_corr

    return new_x, new_y


def calculate_rectangular_magnet_centre_coordinates(x, y, rm_angle):
    """
    The rectangular magnets always have to be 27.2 mm from their circular magnets, and at the appropriate angle
    """
    circular_rectangle_magnet_centre_distance = 27.2

    rectangular_magnet_centre = [
        x + circular_rectangle_magnet_centre_distance * np.cos(rm_angle),
        y + circular_rectangle_magnet_centre_distance * np.sin(rm_angle),
    ]
    return rectangular_magnet_centre


def perform_metrology_calibration(
    input_coords,
    input_theta_d,
    robot_centre,
    robot_shifts_file,
    verbose=True,
    permagnet_theta_corr=True,
    sign="negative",
):
    """
    Apply a correction based on the measured metrology of the robot. Written by Barnaby Norris.
    """

    cent_wrt_origin = np.array(
        [-270.5, 179.0]
    )  # Manufactured offset between metrology origin and plate centre

    # Calculate correction coefficients from metrology
    robot_shifts_abs = np.loadtxt(robot_shifts_file, delimiter=",")
    [x0, y0] = robot_shifts_abs[0, 0:2]
    metr_in_coords = robot_shifts_abs[:, 0:2]
    metr_wanted_coords = np.array([x0, y0]) + robot_shifts_abs[:, 2:4]

    # Solve for transformation coefficients.
    # popt contains the optimised coefficients
    p0 = np.array([0, 0, 0, 1, 1, 0, 0])
    warnings.filterwarnings("ignore", message="Covariance")
    popt, pcov = curve_fit(
        fitting_fun, metr_wanted_coords.reshape((8)), metr_in_coords.reshape((8)), p0=p0
    )

    # # For DEBUG: Measure residuals for the marker measurements
    corrected_data = apply_cal(metr_wanted_coords, popt)
    all_resids = corrected_data - metr_in_coords

    # For backwards compatability reasons, the input coordinates are supplied based on best-known plate centre
    # (given by the 'robot_centre' parameter). Here, we instead want to use the actual measured value of
    # the metrology origin, along with true value (specified to manufacturer) of the offset between the
    # metrology origin and plate centre.
    true_centre = np.array([x0, y0]) + cent_wrt_origin
    input_coords_centred = input_coords - robot_centre + true_centre

    # Apply metrology-based correction
    metr_calibrated_coords = apply_cal(input_coords_centred, popt)

    # Now correct theta, the angle of robot rotation stage. Needs to be rotated by the same amount as global coordinate
    # rotation.
    if permagnet_theta_corr:
        npts = input_coords_centred.shape[0]
        all_theta_ds = np.zeros(npts)
        for k in range(npts):
            dx1 = x0 - input_coords_centred[k, 0]
            dy1 = y0 - input_coords_centred[k, 1]
            phi_old = np.arctan(dy1 / dx1)
            dx2 = x0 - metr_calibrated_coords[k, 0]
            dy2 = y0 - metr_calibrated_coords[k, 1]
            phi_new = np.arctan(dy2 / dx2)
            cur_theta_d = (phi_new - phi_old) / np.pi * 180
            all_theta_ds[k] = cur_theta_d
        theta_d = all_theta_ds
    else:
        theta_d = popt[2]

    if sign == "positive":
        factor = 1
    elif sign == "negative":
        factor = -1
    else:
        raise NameError("Sign must be positive or negative!")
    output_theta_d = (
        input_theta_d + factor * theta_d
    ) % 360  ### SIGN ISSUE: If rotation direction is incorrect, change this + to -

    if verbose:
        if permagnet_theta_corr:
            print(
                f"\tApplied metrology-based calibration and a *per-magnet* theta correction, using the following fitted coefficients: {popt}"
            )
        else:
            print(
                f"\tApplied metrology-based calibration and a *global* theta correction, using the following fitted coefficients: {popt}"
            )

    return metr_calibrated_coords, output_theta_d


def apply_cal(inputs, p):
    # Applies the offset and transformation matrix
    thr = p[2] / 180 * np.pi
    Rmat = np.array([[np.cos(thr), -np.sin(thr)], [np.sin(thr), np.cos(thr)]])
    p[6] = 0  # With 2 shear terms matrix is underconstrained
    shmat = np.array([[1, p[5]], [p[6], 1]])
    sclmat = np.array([[p[3], 0], [0, p[4]]])

    out_coords = np.zeros_like(inputs)
    for k in range(inputs.shape[0]):
        invec = inputs[k, :]
        outvec = invec + np.array([p[0], p[1]])
        outvec = Rmat @ outvec
        outvec = shmat @ outvec
        outvec = sclmat @ outvec
        out_coords[k, :] = outvec
    return out_coords


def apply_cal_no_sh_no_scl(inputs, p):
    # Applies the offset and transformation matrix
    thr = p[2] / 180 * np.pi
    Rmat = np.array([[np.cos(thr), -np.sin(thr)], [np.sin(thr), np.cos(thr)]])
    p[6] = 0  # With 2 shear terms matrix is underconstrained
    shmat = np.array([[1, p[5]], [p[6], 1]])
    sclmat = np.array([[p[3], 0], [0, p[4]]])

    out_coords = np.zeros_like(inputs)
    for k in range(inputs.shape[0]):
        invec = inputs[k, :]
        outvec = invec + np.array([p[0], p[1]])
        outvec = Rmat @ outvec
        # outvec = shmat @ outvec
        # outvec = sclmat @ outvec
        out_coords[k, :] = outvec
    return out_coords


def fitting_fun(Xs, offsX, offsY, theta, sclX, sclY, sh1, sh2):
    params = np.array([offsX, offsY, theta, sclX, sclY, sh1, sh2])
    in_coords = Xs.reshape((4, 2))
    out_coords = apply_cal(in_coords, params)
    out = out_coords.reshape((8))
    return out


def roll_correction(centre_x, centre_y, magnet):
    """
    Calculate the offset which should be applied to the robot x coordinate to account for the roll of the robot arm.
    """
    if magnet == "circular_magnet":
        a = 3.313e-07
        b = -4.507e-05
        c = 1.819e-02
    elif magnet == "rectangular_magnet":
        a = 4.133e-07
        b = -4.151e-05
        c = 1.767e-02

    roll_offset_centre_x = a * centre_y**2 + b * centre_y + c

    return roll_offset_centre_x


def optical_model_correction(df, robot_centre, verbose=True):
    if verbose:
        print("\n\t--> Applying the optical model correction <-- \n")


    # Loop through the hexabundles
    all_hexabundles = df.loc[df["Magnet"] == "circular_magnet", "Hexabundle"].values
    for hexabundle in all_hexabundles:
        circular_magnet = df.loc[
            (df["Hexabundle"] == hexabundle) & (df["Magnet"] == "circular_magnet")
            ]
        rectangular_magnet = df.loc[
            (df["Hexabundle"] == hexabundle) & (df["Magnet"] == "rectangular_magnet")
            ]

        newCirc_x, newCirc_y = calculate_optical_model_correction(
            circular_magnet, robot_centre
        )

        if verbose:
            print( f"\tHexabundle {hexabundle}: \n\tCircular magnet:" )
            print( f"\t\toriginal centre is ({circular_magnet['Center_x'].values[0]:.3f}, "
                   f"{circular_magnet['Center_y'].values[0]:.3f})" )
            print( f"\t\tnew (optical model corrected) centre is ({newCirc_x:.3f}, {newCirc_y:.3f})" )

        # Update the new centres of the circular magnets
        df.at[circular_magnet.index[0], "Center_x"] = newCirc_x
        df.at[circular_magnet.index[0], "Center_y"] = newCirc_y


        # Now find the centres of the rectangular magnets
        angle_for_rectangular_magnet = np.radians(
            270
            - rectangular_magnet["rot_holdingPosition"]
            - rectangular_magnet["rot_platePlacing"]
        ).values[0]

        new_x, new_y = newCirc_x - robot_centre[0], newCirc_y - robot_centre[1] # Subtract the robot center
        [x_rect, y_rect] = calculate_rectangular_magnet_centre_coordinates(
            new_x, new_y, angle_for_rectangular_magnet
        )

        if verbose:
            print( "\tRectangular magnet:" )
            print( f"\t\toriginal centre is ({rectangular_magnet['Center_x'].values[0]:.3f}, "
                   f"{rectangular_magnet['Center_y'].values[0]:.3f})" )
            print( f"\t\tnew centre is ({x_rect + robot_centre[0]:.3f}, {y_rect + robot_centre[1]:.3f})" )

        # Update the new centres of the rectangular magnets
        df.at[rectangular_magnet.index[0], "Center_x"] = x_rect + robot_centre[0]
        df.at[rectangular_magnet.index[0], "Center_y"] = y_rect + robot_centre[1]

    return df


def calculate_optical_model_correction(df_circular, robot_centre):

    def calculate_angles(point1, point2):
        """
        Calculates the angle between the origin (in this case, robot center)
        and a given point, with respect to the +ve x-axis

        anti-clockwise from +ve x-direction to 180-deg gives the angle in +ve radians
        clocwise from +ve x-directions to -180-deg gives the angle in -ve radians
        """
        x1, y1 = point1
        x2, y2 = point2

        dx, dy = x2 - x1, y2 - y1   # calculate the difference in coordinates
        angle = math.atan2(dy, dx)  # the angle in radians

        return angle, np.sign(angle)


    def new_coordinates(magnetPos, robotCenter, Rperpendicular):
        magnetPosX, magnetPosY     = magnetPos
        robotCenterX, robotCenterY = robotCenter

        radial_robotCenter_to_magnet = np.sqrt((robotCenterX - magnetPosX) ** 2.0 + (robotCenterY - magnetPosY) ** 2.0)

        # Solve the equation
        x, y = symbols('x,y')
        eq1 = Eq((robotCenterX - x) ** 2.0 + (robotCenterY - y) ** 2.0, radial_robotCenter_to_magnet ** 2.0)
        eq2 = Eq((magnetPosX - x) ** 2.0 + (magnetPosY - y) ** 2.0, Rperpendicular ** 2.0)

        # Returns two sets of x/y coordinates, either side of the original hexa position
        # (the same Rperpendist away from the initial probe position, either side of the probe)
        result_nonlinear = solve([eq1, eq2], (x, y))

        # Depending on the sign of the "RPerpenDistance_estimate", we need to select the correct coordinate set that
        # "result_nonlinear" returns. To do that, we need to check the angles.
        angle1, sign1 = calculate_angles((robotCenterX, robotCenterY), result_nonlinear[0])
        angle2, sign2 = calculate_angles((robotCenterX, robotCenterY), result_nonlinear[1])

        argmax_angle, argmin_angle = np.argmax([angle1, angle2]), np.argmin([angle1, angle2])

        if sign1 == sign2:  # Either both angles are positive or both negative
            # Rperpendicular < 0: The hexa centers need to move clockwise (in inverted y-axis coordinate system used
            # in quicklook plots) Or anti-clockwise if the y-axis was not inverted.
            if Rperpendicular < 0: new_point = result_nonlinear[argmax_angle]

            # Rperpendicular > 0: The hexa centers need to move anti-clockwise (in inverted y-axis coordinate system used
            # in quicklook plots) Or clockwise if the y-axis was not inverted.
            else: new_point = result_nonlinear[argmin_angle]

        # Special Cases (+ve and -ve angles can happen if hexa position is @0 deg or @180 deg)
        else:
            print(f"Encountered a special case - the angles are +/-ve in radians (i.e. {angle1, angle2})")
            argmin_sign    = np.argmin([sign1, sign2])                               # Which angle is -ve?
            angles_degrees = np.array([math.degrees(angle1), math.degrees(angle2)])  # Angles in degrees (easier to deal with)
            angles_degrees[argmin_sign] = 360. + angles_degrees[argmin_sign]         # Add 360 to -ve angle [deg], so both angles are now positive

            arg_max, arg_min = np.argmax(angles_degrees), np.argmin(angles_degrees)

            # Rperpendicular < 0: Hexa center needs to move clockwise (if y-axis inverted), anti-clockwise if not
            if Rperpendicular < 0:
                if 180. < angles_degrees[arg_max] <= 270.:  # hexa@180 degrees: anti-clockwise move (if y-axis not inverted) --> the largest angle in 3rd quadrant
                    new_point = result_nonlinear[arg_max]
                else:                                       # hexa@0 degrees: anti-clockwise move --> the smallest angle in 1st quadrant
                    new_point = result_nonlinear[arg_min]

            # Rperpendicular > 0: Hexa center needs to move anti-clockwise (if y-axis inverted), clockwise if not
            else:
                if 90. < angles_degrees[arg_min] <= 180.:   # hexa@180 degrees: clockwise move (if y-axis not inverted) --> the smallest angle
                    new_point = result_nonlinear[arg_min]
                else:                                       # hexa@0 degrees: clockwise move --> the largest angles
                    new_point = result_nonlinear[arg_max]

        return new_point, (sign1, sign2)


    # Polynomial Coefficients from the fitting done in 'process_quicklook_files'
    # coeffs = [-5.10579247e-03, -9.08942022e-01, 1.94332118e+02] # Original coeffs from fitting Nov2025 data
    coeffs = [-0.007449708120000001, -0.20090739999999996, 156.2815041] # Original revised with the residual correction seen in Feb2026 data
    polynomial = np.poly1d(coeffs)

    # Polynomial correction is based only on the circular magnet position
    center_x, center_y = df_circular['Center_x'].values[0], df_circular['Center_y'].values[0]

    circProbeX, circProbeY = center_y * 1.0E3, center_x * 1.0E3                 # Convert from mm to microns and switch x/y
    plateX, plateY         = robot_centre[1] * 1.0E3, robot_centre[0] * 1.0E3   # Convert from mm to microns and switch x/y

    radial_robotCenter_to_probe = np.sqrt((plateX - circProbeX) ** 2.0 + (plateY - circProbeY) ** 2.0)

    # Use the polynomial function to estimate the correction
    # Note: In the polynomial coeff estimation, the "radial_plate_to_hexabundle" distance (i.e. radial_robotCenter_to_probe)
    #       was scaled by 1E3, which is applied consistently below.
    RPerpenDistance_estimate = polynomial(radial_robotCenter_to_probe / 1.0E3)

    correctedPos, signs = new_coordinates((circProbeX, circProbeY), (plateX, plateY), RPerpenDistance_estimate)


    return  correctedPos[1] / 1.0E3, correctedPos[0] / 1.0E3  # return the corrected points back in unit of mm (the return order is y/x in robot coordinates)



