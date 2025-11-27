def add_attribute(data, attribute_name, value_func):
    for item in data:
        item[attribute_name] = value_func(item)
    return data


def remove_attribute(data, attribute_name):
    for item in data:
        if attribute_name in item:
            del item[attribute_name]
    return data
