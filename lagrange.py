from scipy.interpolate import lagrange
import numpy
a, b, n = 0, numpy.pi, 100
x_train = numpy.linspace(a, b, n)
y_train = numpy.sin(x_train)
lagrange_model = lagrange(x_train, y_train)
x_test = numpy.linspace(a, b, n)
y_test = numpy.sin(x_test)
train_error = numpy.mean((lagrange_model(x_train) - y_train) ** 2)
test_error = numpy.mean((lagrange_model(x_test) - y_test) ** 2)

print("Train Error initial :", train_error)
print("Test Error initial:", test_error)


def noise(x, sigma):
    noise = numpy.random.normal(loc=0, scale=sigma, size=len(x))
    return x + noise


stdev_values = [0.1, 0.5, 1.0, 2, 4, 8, 15, 20, 25]

for stdev in stdev_values:
    x_t_n = noise(x_train, stdev)
    y_t_n = numpy.sin(x_t_n)

    lagrange_model_noisy = lagrange(x_t_n, y_t_n)
    x_test_n = noise(x_test, stdev)
    y_test_n = numpy.sin(x_test_n)
    train_err_noisy = numpy.mean(
        (lagrange_model_noisy(x_t_n) - y_t_n) ** 2)

    test_err_noisy = numpy.mean(
        (lagrange_model_noisy(x_test_n) - y_test_n) ** 2)

    print(f"\nStandard Deviation: {stdev}")
    print("Train Error Noisy:", train_err_noisy)
    print("Test Error Noisy:", test_err_noisy)
