import torch

COLORMAP = torch.tensor(
    [
        [0.0, 0.0, 0.0],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
        [0.9569, 0.7373, 0.4431],
        [0.4941, 0.0824, 0.7373],
        [0.8667, 0.2627, 0.7216],
        [0.9216, 0.7020, 0.2196],
        [0.1333, 0.8824, 0.0706],
        [0.2745, 0.6392, 0.4471],
        [0.0275, 0.1412, 0.9098],
        [0.4157, 0.6549, 0.0118],
        [0.3294, 0.9765, 0.1804],
        [0.5137, 0.4235, 0.3333],
        [0.6588, 0.3961, 0.1686],
        [0.7451, 0.6863, 0.4078],
        [0.1451, 0.6039, 0.1647],
        [0.2941, 0.0157, 0.7804],
        [0.8353, 0.5961, 0.7176],
        [0.5922, 0.2314, 0.0745],
        [0.5412, 0.2588, 0.4784],
        [0.7882, 0.3176, 0.2392],
        [0.3333, 0.7686, 0.5647],
        [0.6902, 0.3059, 0.2863],
        [0.5294, 0.3294, 0.0706],
        [0.8196, 0.2392, 0.9098],
        [0.3804, 0.8941, 0.6784],
        [0.4706, 0.5373, 0.1098],
        [0.1686, 0.5647, 0.5922],
        [0.1098, 0.9020, 0.1451],
        [0.0941, 0.8196, 0.0863],
        [0.3882, 0.1804, 0.4941],
        [0.8196, 0.4784, 0.7725],
        [0.5843, 0.5255, 0.6275],
        [0.0, 1.0, 1.0],
        [0.8667, 0.0471, 0.4000],
        [0.6353, 0.6980, 0.0902],
        [0.5137, 0.8000, 0.5216],
        [0.1569, 0.3529, 0.1882],
        [0.8627, 0.4353, 0.4392],
        [0.1686, 0.0039, 0.5098],
        [0.0, 1.0, 0.0],
        [0.2314, 0.6078, 0.2078],
        [0.5765, 0.0510, 0.8118],
        [0.7529, 0.3882, 0.8471],
        [0.8471, 0.5255, 0.2510],
        [0.0, 0.0, 1.0],
        [0.7843, 0.5412, 0.2824],
        [0.7176, 0.4549, 0.8588],
        [1.0, 1.0, 1.0],
        [0.6863, 0.1843, 0.0157],
        [0.4980, 0.0118, 0.2431],
        [1.0, 0.0, 0.0],
        [0.4471, 0.1882, 0.3686],
        [0.6078, 0.5529, 0.4902],
        [0.6863, 0.6941, 0.2941],
        [1.0, 0.0, 1.0],
        [0.2471, 0.8235, 0.9137],
        [0.0667, 0.8392, 1.0000],
        [1.0, 1.0, 0.0],
        [0.2392, 0.1882, 0.9059],
    ]
)
