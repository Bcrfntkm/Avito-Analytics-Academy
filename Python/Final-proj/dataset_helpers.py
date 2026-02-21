def filter_by_condition(data, condition_func):
    return [item for item in data if condition_func(item)]


def group_by_attributes(data, attributes):
    result = {}
    for item in data:
        key = tuple(item.get(attr) for attr in attributes)
        if key not in result:
            result[key] = []
        result[key].append(item)
    return result


def apply_aggregation(data, aggregation_func):
    return aggregation_func(data)
