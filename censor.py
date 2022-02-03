import cv2
import numpy as np
import math
import json

def get_distance(a,b, x,y):
    return math.sqrt((a-x)**2 + (b-y)**2)

def get_tuple(arr):
    result_arr = []
    for i in range(len(arr[0])):
        result_arr.append((arr[0][i], arr[1][i]))
    return result_arr

def get_2d_array(tup):
    result_arr = [[],[]]
    for item in tup:
        result_arr[0].append(item[0])
        result_arr[1].append(item[1])
    return result_arr

def get_ranged_averages(arr_of_locations, threshold):
    result_arr = []
    result_arr.append(arr_of_locations[0])
    for elem in arr_of_locations:
        if elem in result_arr:
            continue
        min_distance = 100
        for item in result_arr:
            distance = get_distance(elem[0], elem[1], item[0], item[1])
            #print(elem, item, distance)
            if distance < min_distance:
                min_distance = distance
        if min_distance > threshold:
            result_arr.append(elem)
            print(result_arr, min_distance)
    return get_2d_array(result_arr)


def temp_match(files_array, template_name, confidence_threshold, distance_threshold=10):
    files = files_array

    template = cv2.imread(template_name)
    h, w, _ = template.shape

    for name in files:
        img = cv2.imread(name)
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        #print(result[0][0])
        arr = np.where(result >= confidence_threshold)
        #print(get_ranged_averages(arr[0], 10))
        #print(get_tuple(arr))
        newarr = (get_ranged_averages(get_tuple(arr), distance_threshold))
        print(get_tuple(arr)==get_tuple(newarr))
        for i in range(len(newarr[0])):
            y, x = newarr[0][i], newarr[1][i]
            conf = result[y][x]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            text = f'Confidence: {round(float(conf), 2)}'
            cv2.putText(img, text, (x, y), 1, cv2.FONT_HERSHEY_PLAIN, (0, 0, 0), 2)
        cv2.imshow(name, img)
        
    cv2.imshow('Template', template)
    cv2.waitKey(0)

def confidence(img, template):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    conf = res.max()
    return np.where(res == conf), conf

def main():
    with open('config.json') as f:
        data = json.load(f)
        temp_match(data["imgs"], data["template"],0.4, 30)

if __name__ == "__main__":
    main()