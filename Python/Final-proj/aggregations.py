import statistics


def calculate_mean(data, field):
    values = [item[field] for item in data if field in item and item[field] is not None]
    return statistics.mean(values) if values else None


def calculate_median(data, field):
    values = [item[field] for item in data if field in item and item[field] is not None]
    return statistics.median(values) if values else None


def calculate_max(data, field):
    values = [item[field] for item in data if field in item and item[field] is not None]
    return max(values) if values else None


def get_top_n(data, field, n=10):
    valid_data = [item for item in data if field in item and item[field] is not None]
    return sorted(valid_data, key=lambda x: x[field], reverse=True)[:n]
