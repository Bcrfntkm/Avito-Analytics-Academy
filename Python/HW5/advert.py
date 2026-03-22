import json
import keyword


class ColorizeMixin:
    repr_color_code = 33

    def __repr__(self):
        text = super().__repr__()
        return f"\033[{self.repr_color_code}m{text}\033[0m"


class JSONObject:
    def __init__(self, mapping):
        for key, value in mapping.items():
            if keyword.iskeyword(key):
                key += '_'
            
            if isinstance(value, dict):
                setattr(self, key, JSONObject(value))
            else:
                setattr(self, key, value)


class Advert(ColorizeMixin):
    repr_color_code = 33
    
    def __init__(self, mapping):
        if 'title' not in mapping:
            raise ValueError("title is required")
        
        if 'price' not in mapping:
            mapping['price'] = 0
            
        self._price = None
        self.price = mapping['price']
        
        for key, value in mapping.items():
            if key == 'price':
                continue
                
            if keyword.iskeyword(key):
                key += '_'
            
            if isinstance(value, dict):
                setattr(self, key, JSONObject(value))
            else:
                setattr(self, key, value)
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("price must be >= 0")
        self._price = value
    
    def __repr__(self):
        return f"{self.title} | {self.price} ₽"