import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import streamlit.components.v1 as components
from distutils.command.build import build

# --- Global constants ---
g = 9.81
penalty_coeff = 0.02  # Tune as needed
_RELEASE = True

# --- Define the simulation function ---
def simulate_flight(launch_angle, nose_angle_init, RPM_init, wind_speed, KE, gravity, radius, area, rho, mass):
    moment_of_inertia = 0.5 * mass * radius ** 2
    omega = 2 * math.pi * RPM_init / 60

    angle_diff = abs(launch_angle - nose_angle_init)
    energy_penalty_factor = max(0, 1 - penalty_coeff * angle_diff)
    effective_KE = KE * energy_penalty_factor

    if effective_KE <= 0:
        return 0, [], []

    rotational_KE = 0.5 * moment_of_inertia * omega ** 2
    translational_KE = effective_KE - rotational_KE
    if translational_KE <= 0:
        return 0, [], []

    launch_speed = math.sqrt((2 / mass) * translational_KE)
    if launch_speed <= 0:
        return 0, [], []

    launch_angle_rad = math.radians(launch_angle)
    nose_angle_rad = math.radians(nose_angle_init)
    omega_nose = 0.0
    x, y = 0, 1
    vx = launch_speed * math.cos(launch_angle_rad)
    vy = launch_speed * math.sin(launch_angle_rad)
    wind_vx = wind_speed
    wind_vy = 0

    dt = 0.01
    t = 0
    max_sim_time = 30
    decay_rate = 0.5
    k_torque = 20.0
    max_distance = 0

    # Store trajectory
    x_coords = [x]
    y_coords = [y]

    while y > 0 and t < max_sim_time:
        v_rel_x = vx - wind_vx
        v_rel_y = vy - wind_vy
        airflow_angle = math.atan2(v_rel_y, v_rel_x)

        omega -= decay_rate * dt
        if omega < 0:
            omega = 0

        current_RPM = omega * 60 / (2 * math.pi)
        damping_factor = 2.5 * (1 - math.exp(-current_RPM / 500))
        stability_factor = 1 - math.exp(-current_RPM / 300)

        aoa = nose_angle_rad - airflow_angle
        torque = -k_torque * aoa
        alpha = torque / moment_of_inertia
        omega_nose += alpha * dt
        omega_nose -= damping_factor * omega_nose * dt
        nose_angle_rad += omega_nose * dt

        aoa_deg = math.degrees(nose_angle_rad - airflow_angle)

        if abs(aoa_deg) < 15:
            Cl_fake = 0.15 + 1.4 * math.radians(aoa_deg)
        else:
            if aoa_deg > 0:
                Cl_fake = max(0, 0.15 + 1.4 * math.radians(15) * (1 - (aoa_deg - 15) / 15))
            else:
                Cl_fake = min(0, 0.15 + 1.4 * math.radians(-15) * (1 - (-aoa_deg - 15) / 15))

        Cd_fake = 0.08 + 2 * Cl_fake ** 2
        Cl = Cl_fake * stability_factor
        Cd = Cd_fake * (1 + 3 * (1 - stability_factor))

        v = math.sqrt(v_rel_x ** 2 + v_rel_y ** 2)
        if v == 0:
            break

        Flift = 0.5 * rho * area * Cl * v ** 2
        Fdrag = 0.5 * rho * area * Cd * v ** 2

        Flx = -Flift * (v_rel_y / v)
        Fly = Flift * (v_rel_x / v)
        Fdx = -Fdrag * (v_rel_x / v)
        Fdy = -Fdrag * (v_rel_y / v)

        Fx = Fdx + Flx
        Fy = Fdy + Fly - mass * gravity

        ax = Fx / mass
        ay = Fy / mass

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        x_coords.append(x)
        y_coords.append(y)

        t += dt
        max_distance = max(max_distance, x)

    return max_distance, x_coords, y_coords


# --- Streamlit App ---
st.title("Optimizing Frisbee Flight")
st.subheader("This app simulates the physics of a flying frisbee using a computational model based on real-world aerodynamics. Users can adjust parameters such as spin rate (RPM), release angle, velocity, and wind speed to visualize how each factor influences frisbee flight. The model incorporates lift and drag coefficients (Cl and Cd), gyroscopic stability, tilt, and wind vectors to generate accurate flight paths.")
st.markdown("made by Seiji Iigaya. Check out my frisbee team website: https://www.tigerultimatenj.com/ Any suggestions or feedback? Contact me at seijithestone@gmail.com")
st.sidebar.title("Sidebar Title")
st.sidebar.markdown("Sliders")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", -10.0, 10.0, 0.0, 0.1, help="Meters per second. Positive wind speeds are tailwinds while negative wind speeds are headwinds.")
KE = st.sidebar.slider("Initial Kinetic Energy (J)", 10.0, 120.0, 20.0, 0.5, help="Joules. The maximum energy input. As a benchmark, this will be about 10 for middle school, 20 for high school, 35 for college, and up to 80 for professional players.")
gravity = st.sidebar.slider("Gravity (m/s²)", 8.0, 12.0, 9.81, 0.01, help="Usually you are on Earth, so no need to change this.")
radius = st.sidebar.slider("Frisbee Radius (m)", 0.05, 0.2, 0.136525, 0.005, help = "This is the radius of the disc in meters")
area = st.sidebar.slider("Frisbee Area (m²)", 0.01, 0.1, 0.018639, 0.001, help="Area of disc in square meters")
rho = st.sidebar.slider("Air Density (kg/m³)", 0.5, 1.5, 1.225, 0.01, help="Air density, kg/m^3. The default is 15 degrees at see level")
mass = st.sidebar.slider("Frisbee Mass (kg)", 0.05, 0.2, 0.175, 0.005, help="weight of disc in kilograms")
physics_description = """
This simulation models the real-world flight of a frisbee using key principles of aerodynamics and rotational dynamics:

- **Translational Kinetic Energy**  
  $KE = \\frac{1}{2} m v^2$  
  Determines the initial forward motion of the disc, where:
  - $m$ = mass of the frisbee (≈ 0.175 kg)  
  - $v$ = release velocity (m/s)

- **Rotational Energy**  
  $KE_{rot} = \\frac{1}{2} I \\omega^2$  
  Where:
  - $I = \\frac{1}{2} m r^2$ is the moment of inertia  
  - $\\omega$ is angular velocity (rad/s)  
  Higher spin increases flight stability.

- **Lift and Drag Forces**  
  - Lift: $F_L = \\frac{1}{2} C_L \\rho A v^2$  
  - Drag: $F_D = \\frac{1}{2} C_D \\rho A v^2$  
  Where:
  - $C_L$, $C_D$ are lift and drag coefficients varying with angle of attack  
  - $\\rho$ is air density (~1.225 kg/m³)  
  - $A$ is the frisbee's reference area

- **Gyroscopic Stability**  
  Modeled via spin rate (RPM) and precession; higher spin provides more stable flight.

- **Wind Effects**  
  Headwinds and tailwinds affect relative airspeed, modifying aerodynamic forces dynamically.

- **Tilt Dynamics**  
  Aerodynamic torque causes the frisbee's nose angle to adjust during flight, affecting lift and drag.

---

This model assumes that the disc is circular, it does not account for the rim of the disc.
"""


with st.expander("🔬 Physics Model Details (click to expand)"):
    st.markdown(physics_description)





if st.button("Find Optimal Settings"):
    with st.spinner("Calculating optimal frisbee settings...approximately 20 seconds"):
        best_range = 0
        best_settings = (0, 0, 0)
        best_trajectory = ([], []) # Store x and y coordinates
        for angle in range(5, 46, 2):
            for nose in range(5, 46, 2):
                for rpm in range(500, 4001, 30):
                    dist, x_coords, y_coords = simulate_flight(angle, nose, rpm, wind_speed, KE, gravity, radius, area, rho, mass)
                    if dist > best_range:
                        best_range = dist
                        best_settings = (angle, nose, rpm)
                        best_trajectory = (x_coords, y_coords)

    st.success(f"Max Distance:  {best_range:.2f} m")
    st.info(f"Best Launch Angle:  {best_settings[0]}°\nBest Nose Angle:  {best_settings[1]}°\nBest RPM: {best_settings[2]}")

    # now add the code to plot the trajectory of the frisbee
    fig, ax = plt.subplots()
    ax.plot(best_trajectory[0], best_trajectory[1])
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("height (m)")
    ax.set_title("Frisbee Trajectory")
    st.pyplot(fig)
