import geopy.distance

class Point:
    def __init__(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude
    
    def distance(a, b):
        return geopy.distance.distance((a.latitude, a.longitude), (b.latitude, b.longitude))