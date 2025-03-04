import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

# Define fuzzy logic variables
size = np.arange(0, 101, 1)
speed = np.arange(0, 101, 1)
class_validity = np.arange(0, 101, 1)

# Fuzzy membership functions for size
size_small = fuzz.trimf(size, [0, 0, 50])
size_medium = fuzz.trimf(size, [0, 50, 100])
size_large = fuzz.trimf(size, [50, 100, 100])

# Fuzzy membership functions for speed
speed_slow = fuzz.trimf(speed, [0, 0, 50])
speed_moderate = fuzz.trimf(speed, [0, 50, 100])
speed_fast = fuzz.trimf(speed, [50, 100, 100])

# Fuzzy membership functions for class validity
class_validity_low = fuzz.trimf(class_validity, [0, 0, 50])
class_validity_medium = fuzz.trimf(class_validity, [0, 50, 100])
class_validity_high = fuzz.trimf(class_validity, [50, 100, 100])

# Plot size membership functions
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.plot(size, size_small, label='Small')
plt.plot(size, size_medium, label='Medium')
plt.plot(size, size_large, label='Large')
plt.title('Size Membership Functions')
plt.xlabel('Size')
plt.ylabel('Membership')
plt.legend()

# Plot speed membership functions
plt.subplot(132)
plt.plot(speed, speed_slow, label='Slow')
plt.plot(speed, speed_moderate, label='Moderate')
plt.plot(speed, speed_fast, label='Fast')
plt.title('Speed Membership Functions')
plt.xlabel('Speed')
plt.ylabel('Membership')
plt.legend()

# Plot class validity membership functions
plt.subplot(133)
plt.plot(class_validity, class_validity_low, label='Low')
plt.plot(class_validity, class_validity_medium, label='Medium')
plt.plot(class_validity, class_validity_high, label='High')
plt.title('Class Validity Membership Functions')
plt.xlabel('Class Validity')
plt.ylabel('Membership')
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
