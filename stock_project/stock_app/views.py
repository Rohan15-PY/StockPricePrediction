from django.shortcuts import render
from .ml_model import predict_price, train_model

def home(request):
    return render(request, "index.html")

def predict(request):
    if request.method == "POST":
        stock_name = request.POST.get("stock_name")

        try:
            price = predict_price(stock_name)
        except:
            # If model not trained yet, train now
            train_model(stock_name)
            price = predict_price(stock_name)

        return render(request, "result.html", {
            "stock_name": stock_name,
            "price": round(float(price), 2)
        })
