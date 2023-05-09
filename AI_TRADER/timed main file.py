import timedelta as timedelta
from keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
import alpaca_trade_api as tradeapi
import time
from datetime import datetime, timedelta
import threading
from Api_Passwords import Api_Key, Api_Secret
API_KEY = Api_Key
API_SECRET = Api_Secret
BASE_URL = 'https://paper-api.alpaca.markets'
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
TICKER = 'AAPL'
SYMBOL = 'AAPL'


def is_time_between(start, end):
    now = datetime.now().time()
    if start < end:
        return start <= now <= end
    else:
        return start <= now or now <= end

def main_program():
    def get_data(symbol, start_date, end_date):
        df = yf.download(symbol, start=start_date, end=end_date, interval="1m")
        print(df.tail())
        return df

    def preprocess_data(df, feature_cols, target_col):
        data = df[feature_cols]
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X = data_scaled[:-1]
        y = df[target_col].values[1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, scaler

    def train_svm(X_train, y_train):
        model = SVR(kernel='rbf')
        model.fit(X_train, y_train)
        return model

    def train_ann(X_train, y_train):
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        return model

    def trade(symbol, quantity, prediction, current_price, api):
        position_exists = False
        try:
            position = api.get_position(symbol)
            position_exists = True
        except Exception as e:
            print("Error:", e)
            print("No existing position found for symbol:", symbol)
        if prediction > current_price and not position_exists:
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        elif prediction < current_price and position_exists:
            api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )

    start_date = '2017-01-01'
    today = datetime.now()
    end_date = (today - timedelta(days=6)).strftime('%Y-%m-%d')
    quantity = 10
    df = get_data(SYMBOL, end_date, today)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, ['Close'], 'Close')
    svm_model = train_svm(X_train, y_train)
    ann_model = train_ann(X_train, y_train)
    print('models have been trained')

    # Save the SVM model
    joblib.dump(svm_model, 'svm_model_aapl.joblib')

    # Save the ANN model
    ann_model.save('ann_model_aapl.h5')

    # Load the saved SVM model
    loaded_svm_model = joblib.load('svm_model_aapl.joblib')

    # Load the saved ANN model
    loaded_ann_model = load_model('ann_model_aapl.h5')
    print("models have been loaded")
    # Make predictions using the loaded SVM and ANN models
    predictions_svm = loaded_svm_model.predict(X_test)
    print("predict svm")
    predictions_ann = loaded_ann_model.predict(X_test).flatten()
    print('predict ann')
    predictions_combined = (predictions_svm + predictions_ann) / 2
    print('prediction combined')

    def evaluate_performance(api, initial_balance):
        account = api.get_account()
        current_balance = float(account.equity)
        profit = current_balance - initial_balance
        print("performance evaluating")
        return profit

    repitition = 0
    while True:
        def main():
            print(f"Repetition: {repitition + 1}")

            start_date = '2017-01-01'
            today = datetime.now()
            end_date = (today - timedelta(days=6)).strftime('%Y-%m-%d')
            quantity = 10
            initial_balance = float(api.get_account().equity)
            time_to_wait = 60  # 1 minute
            train_event.wait()
            for i in range(len(predictions_combined)):
                prediction = predictions_combined[i]
                current_price = df['Close'].iloc[-len(predictions_combined) + i]
                trade(TICKER, quantity, prediction, current_price, api)
                print("trade ran for: " + str(TICKER))
                time.sleep(time_to_wait)
                print('trades are executing')

            profit = evaluate_performance(api, initial_balance)
            print(f"Profit: ${profit}")
            train_event.clear()

        def sleep():
            start_date = '2017-01-01'
            today = datetime.now()
            end_date = (today - timedelta(days=6)).strftime('%Y-%m-%d')
            quantity = 10
            df = get_data(SYMBOL, end_date, today)
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df, ['Close'], 'Close')
            svm_model = train_svm(X_train, y_train)
            ann_model = train_ann(X_train, y_train)
            print('models have been trained')

            # Save the SVM model
            joblib.dump(svm_model, 'svm_model_aapl.joblib')

            # Save the ANN model
            ann_model.save('ann_model_aapl.h5')

            # Load the saved SVM model
            loaded_svm_model = joblib.load('svm_model_aapl.joblib')

            # Load the saved ANN model
            loaded_ann_model = load_model('ann_model_aapl.h5')
            print("models have been loaded")
            # Make predictions using the loaded SVM and ANN models
            predictions_svm = loaded_svm_model.predict(X_test)
            print("predict svm")
            predictions_ann = loaded_ann_model.predict(X_test).flatten()
            print('predict ann')
            predictions_combined = (predictions_svm + predictions_ann) / 2
            print('prediction combined')
            train_event.set()

        train_event = threading.Event()

        main_thread = threading.Thread(target=main)
        sleep_thread = threading.Thread(target=sleep)

        # Start the threads
        main_thread.start()
        sleep_thread.start()

        # Wait for both threads to finish
        main_thread.join()
        sleep_thread.join()



trading_start_time = datetime.strptime("09:30", "%H:%M").time()
trading_end_time = datetime.strptime("16:00", "%H:%M").time()

if not is_time_between(trading_start_time, trading_end_time):
    time_to_wait = (datetime.combine(datetime.today(), trading_start_time) - datetime.now()).seconds
    print(f"Waiting for {time_to_wait / 60} minutes until 9:30 AM.")
    time.sleep(time_to_wait)

main_program()