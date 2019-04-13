# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import dialogflow_v2beta1 as dialogflow
import os
import json

import predict_diagnosis

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./test-agent-4fd0f3213667.json"
app = Flask(__name__)
CORS(app)
valid_answers = ["Зуд",
                 "кожная_сыпь",
                 "высыпания_на_коже",
                 "непрерывное_чихание",
                 "дрожь",
                 "озноб",
                 "боль_в_суставах",
                 "боль_в_животе",
                 "изжога",
                 "язвы_на_языке",
                 "атрофия_мышц",
                 "рвота",
                 "затрудненное_мочеиспускание",
                 "spotting_ urination",
                 "усталость",
                 "увеличение_веса",
                 "тревога",
                 "холодные_руки_и_ноги",
                 "перепады_настроения",
                 "потеря_веса",
                 "беспокойство",
                 "вялость",
                 "пятна_на_горле",
                 "нерегулярный_уровень_сахара",
                 "кашель",
                 "высокая_температура",
                 "запавшие_глаза",
                 "одышки",
                 "повышенное_потоотделение",
                 "обезвоживание",
                 "расстройство_желудка",
                 "головная_боль",
                 "желтоватая_кожа",
                 "темная_моча",
                 "тошнота",
                 "потеря_аппетита",
                 "боль_за_глазами",
                 "боль_в_спине",
                 "запор",
                 "боль_в_животе",
                 "понос",
                 "слабая_лихорадка",
                 "желтая_моча",
                 "пожелтение_глаз",
                 "острая_печеночная_недостаточность",
                 "избыток_жидкости",
                 "отек_желудка",
                 "опухшие_лимфатические_узлы",
                 "недомогание",
                 "затуманенное_и_искаженное_зрение",
                 "мокрота",
                 "раздражение_горла",
                 "покраснение_глаз",
                 "давление_в_пазухах",
                 "насморк",
                 "запор",
                 "грудная_боль",
                 "слабость_в_конечностях",
                 "высоки_пульс",
                 "боль_при_дефракции",
                 "боль_в_анальной_области",
                 "кровавый_стул",
                 "раздражение_в_анусе",
                 "боль_в_шее",
                 "головокружение",
                 "судороги",
                 "кровоподтеки",
                 "ожирение",
                 "опухшие_ноги",
                 "опухшие_кровеносные_сосуды",
                 "опухшее_лицо_и_глаза",
                 "увеличенная_щитовидная_железа",
                 "ломкие_ногти",
                 "опухшие_конечности",
                 "чрезмерный_голод",
                 "случаные_половые_контакты",
                 "сухие_губы",
                 "невнятная_речь",
                 "боль_в_колене",
                 "боль_в_тазобедренном_суставе",
                 "мышечная_слабость",
                 "ригидность_затылочных_мышц",
                 "отек_суставов",
                 "жесткость_движения",
                 "вращающиеся_движения",
                 "потеря_баланса",
                 "шаткость",
                 "слабость_одной_стороны_тела",
                 "потеря_обоняния",
                 "дискомфорт_мочевого_пузыря",
                 "foul_smell_of urine",
                 "недержание_мочи",
                 "недержание_газов",
                 "внутренний_зуд",
                 "Токсичный_вид_(тиф)",
                 "депрессия",
                 "раздражительность",
                 "боль_в_мышцах",
                 "измененный_сенсорий",
                 "красные_пятна_на_теле",
                 "боль_в_животе",
                 "ненормальная_менструация",
                 "dischromic _patches",
                 "поливать_из_глаз",
                 "повышенный_аппетит",
                 "Полиурия",
                 "история_семьи",
                 "вязкая_мокрота",
                 "рыжая_мокрота",
                 "притупленная_концентрации",
                 "нарушения_зрения",
                 "получение_переливания_крови",
                 "получение_нестерильных_инъекций",
                 "Кома",
                 "желудочное_кровотечение",
                 "вздутие_живота",
                 "история_употребления_алкоголя",
                 "перегрузка_жидкостью",
                 "кровь_в_мокроте",
                 "заметные_вены_на_голени",
                 " учащенное сердцебиение",
                 "болезненная_ходьба",
                 "гнойные_прыщи",
                 "угри",
                 "снующий",
                 "пилинг_кожи",
                 "серебро_как_пыль",
                 "повреждения_ногтей",
                 "воспаленные_ногти",
                 "мозоль",
                 "краснота_вокруг_носа",
                 "желтые_слизистые_выделения"]

piluli_sheet = pd.read_csv('piluli_sheet.csv',encoding = 'utf8')


@app.route('/id_parse', methods=['POST', 'GET'])
def id_parse():
    if request.method == 'POST':
        infection = request.data['Infection']
        string_id = piluli_sheet[piluli_sheet['Infection'] == infection]
        myDict = pd.DataFrame(string_id)
        return json.dumps(myDict.to_dict(orient='records'), ensure_ascii=False)


@app.route('/test_with_key', methods=['POST', 'GET'])
def index():
    data = {
        "treatments": [],
        "drugs": [],
        "tests": []
    }
    data_req = json.loads(request.data)
    input_text = data_req['user_question']
    if (len(data_req['symptoms']) > 0):
        symptoms = data_req['symptoms']
    else:
        symptoms = ''

    if input_text and len(symptoms) == 0:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path("test-agent-3c1d1", "5f0f6fa22a46436c9695cf8333dac911")

        text_input = dialogflow.types.TextInput(text=input_text, language_code='ru')
        query_input = dialogflow.types.QueryInput(text=text_input)
        final = session_client.detect_intent(session=session, query_input=query_input)
        print(final.query_result.fulfillment_text)
        if final.query_result.fulfillment_text in valid_answers:
            symptoms = dict()
            txt = final.query_result.intent.display_name
            print(txt)
            postiton1=txt.find('{')
            position2=txt.find('}')
            txt = txt[postiton1+1:]
            symptoms[txt] = 1
        else:
            data['simple_text'] = final.query_result.fulfillment_text
            data['symptoms'] = {}
            return json.dumps(data, ensure_ascii=False)

    print(symptoms)
    prediction_result = predict_diagnosis.predict(symptoms)
    print(prediction_result)
    if 'error' in prediction_result:
        return prediction_result

    if 'question' in prediction_result:
        data['question'] = prediction_result['question']
        data['question_ref'] = prediction_result['question_ref']
        data['symptoms'] = symptoms
        return json.dumps(data, ensure_ascii=False)
    else:
        # model_response = requests.post('model', data={'final': str(final)})
        print(prediction_result['diagnosis'])
        if prediction_result['diagnosis'] in list(piluli_sheet['Infection']):  # example
            filtered_sheet = piluli_sheet[piluli_sheet['Infection'] == prediction_result['diagnosis']]
            print(filtered_sheet)
            data["infection"] = str(filtered_sheet['Инфекция'].values[0])

            treatments = ['Лечение1', 'Лечение2', 'Лечение3']
            for treatment in treatments:
                if str(filtered_sheet[treatment].values[0]) != 'nan':
                    data['treatments'].append(str(filtered_sheet[treatment].values[0]))

            drugs = ['Препарат1', 'Препарат2', 'Препарат3']
            for drug in drugs:
                if str(filtered_sheet[drug].values[0]) != 'nan':
                    data['drugs'].append(str(filtered_sheet[drug].values[0]))

            analyzes = ['Анализ1', 'Анализ2', 'Анализ3']
            for analysis in analyzes:
                if str(filtered_sheet[analysis].values[0]) != 'nan':
                    data['tests'].append(str(filtered_sheet[analysis].values[0]))

            data['doctor'] = str(filtered_sheet['Врач'].values[0])
            return json.dumps(data, ensure_ascii=False)
        else:
            return "{error: 'wrong data'}"


def test():
    # data = {
    #     "infection": '',
    #     "treatments": [],
    #     "drugs": [],
    #     "tests": [],
    #     "doctor": ''
    # }
    data = {}
    input_text = 'у меня болит живот'
    symptoms = ''

    if input_text and len(symptoms) == 0:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path("test-agent-3c1d1", "5f0f6fa22a46436c9695cf8333dac911")

        text_input = dialogflow.types.TextInput(text=input_text, language_code='ru')
        query_input = dialogflow.types.QueryInput(text=text_input)
        final = session_client.detect_intent(session=session, query_input=query_input)
        print(final.query_result.fulfillment_text)
        if final.query_result.fulfillment_text in valid_answers:
            symptoms = dict()
            txt = final.query_result.intent.display_name
            txt = txt[txt.find('{') + 1:txt.rfind('}')]
            symptoms[txt] = 1
        else:
            data['simple_text'] = final.query_result.fulfillment_text
            data['symptoms'] = {}
            return json.dumps(data, ensure_ascii=False)

    prediction_result = predict_diagnosis.predict(symptoms)
    if 'question' in prediction_result:
        data['question'] = prediction_result['question']
        data['question_ref'] = prediction_result['question_ref']
        data['symptoms'] = symptoms
        return json.dumps(data, ensure_ascii=False)
    else:
        # model_response = requests.post('model', data={'final': str(final)})
        if prediction_result.value in list(piluli_sheet['Infection']):  # example
            filtered_sheet = piluli_sheet[piluli_sheet['Infection'] == prediction_result.value]

            data["infection"] = filtered_sheet['Инфекция'][0]

            treatments = ['Лечение1', 'Лечение2', 'Лечение3']
            for treatment in treatments:
                if isinstance(filtered_sheet[treatment][0], str):
                    data['treatments'].append(filtered_sheet[treatment][0])

            drugs = ['Препарат1', 'Препарат2', 'Препарат3']
            for drug in drugs:
                if isinstance(filtered_sheet[drug][0], str):
                    data['drugs'].append(filtered_sheet[drug][0])

            analyzes = ['Анализ1', 'Анализ2', 'Анализ3']
            for analysis in analyzes:
                if isinstance(filtered_sheet[analysis][0], str):
                    data['tests'].append(filtered_sheet[analysis][0])

            data['doctor'] = filtered_sheet['Врач'][0]
            return json.dumps(data, ensure_ascii=False)
        else:
            return "{error: 'wrong data'}"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port='3389', debug=False)
