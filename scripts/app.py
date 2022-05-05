from flask import Flask, redirect, url_for, request,jsonify
import os
import spamfilter as sf
app = Flask(__name__)

@app.route('/result',methods=['GET', 'POST'])
def login():

    if request.method == 'POST':
        res = request.form
        output, prob = sf.predict_in(res['message'])
        if(res['status']=='check'):
            print(output, prob*100)
            return jsonify({'response':output,'p_spam':prob[0][0],'p_ham':prob[0][1]}),200
        else:
            print(output, prob*100)
            return jsonify({'response':output,'p_spam':prob[0][0],'p_ham':prob[0][1]}),200

    else:
        print("Bad Request")

if __name__ == '__main__':
    app.run(debug = True)







# If you use Google Chrome browser you can hack with an extension.

# You can find a Chrome extension(https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi) that will modify CORS headers on the fly in your application. Obviously, this is Chrome only, but I like that it works with zero changes anywhere at all.

# You can use it for debugging your app on a local machine (if everything works in production).

# Notice: If URL becomes broken the extension name is Access-Control-Allow-Origin: *. I recommend you to disable this extension when you not working on your stuff, because, for example, youtube does not work with this extension.