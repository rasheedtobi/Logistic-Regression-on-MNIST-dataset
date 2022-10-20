
import model *
import matplotlib.pyplot as plt
#%matplotlib inline


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

def test_predict(datast, indx)    
    img, label = datast[indx]
    plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, model)