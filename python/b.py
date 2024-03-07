def mysum(a, b):
    return a + b


x = mysum(10, 15)
print("Hello world from b.py")


def get_tuple_shape(t):
    if isinstance(t, tuple):
        return (len(t),) + get_tuple_shape(t[0])
    else:
        return ()


# 示例：嵌套元组
nested_tuple = ((1, 2), (3, 4), (5, 6))

# 打印嵌套元组的“shape”
print(get_tuple_shape(nested_tuple))
