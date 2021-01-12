var express = require('express');
var multer = require('multer');
var app = express();
var iconv  = require('iconv-lite');
const mongoose = require('mongoose');
var store = require('./Schema/StoreModel.js');
var menu = require('./Schema/MenuModel.js');
var done = false;

mongoose.connect('mongodb://127.0.0.1:27017/for_graduation'); //db 연결
var connection = mongoose.connection; //몽구스 연결
connection.on('error', console.error.bind(console, 'connection error:'));
var api_url = 'https://openapi.naver.com/v1/papago/n2mt';
var request = require('request');
var client_id = 'M1XICOLgvbvOqH2X6OgX';
var client_secret = 'HThhehJrxR';
app.use(multer({
    dest: './uploads/',
    rename: function (fieldname, filename) {
        return Date.now();
    },
    onFileUploadStart: function (file) {
        console.log(file.originalname + ' is starting ...')
    },
    onFileUploadComplete: function (file) {
        console.log(file.fieldname + ' uploaded to  ' + file.path)
        var spawn = require("child_process").spawn;
        var process = spawn('python', ['OCR_Text_Detection.py', file.path]);
        process.stdout.on('data', function (data) {
            console.log("로그- " + data.toString());
        });
        done = true;
    }
}));

app.get('/', function (req, res) {
    res.sendfile('index.html');
});

app.post('/api/photo', function (req, res) {
    if (done == true) {
        console.log(req.files);

        var spawn = require("child_process").spawn;
        var process = spawn('python', ['OCR_result.py']);
        var result='';
        iconv.extendNodeEncodings();
        process.stdout.on('data', function (data2) {
            result = iconv.decode(data2, 'EUC-KR').toString()
            result = result.replace(/\r\n/g, '');
            console.log("로그- " + result);
            store.find({'Store_Name':{'$regex':result, '$options':'igm'}}).limit(5).exec(function(err, store_data){
              menu.find({'Menu_ko_Name':{'$regex':result, '$options':'igm'}}).limit(5).exec(function(err, menu_data){
                //console.log(menu_data)
                var options = {
                    url: api_url,
                    form: {'source':'ko', 'target':'en', 'text':result},
                    headers: {'X-Naver-Client-Id':client_id, 'X-Naver-Client-Secret': client_secret}
                 };
                request.post(options, function (error, response, body) {
                  if (!error && response.statusCode == 200) {
                    var objBody = JSON.parse(response.body);
                    console.log(objBody.message.result.translatedText);
                    var result_list = {
                      Result: result,
                      Store_Data:store_data,
                      Menu_Data:menu_data,
                      Translation_Data:objBody.message.result.translatedText
                    }
                    var jsondata = JSON.stringify(result_list);
                    //res.end("Store Result: \n" + store_data + "\n"+"Menu Result: \n" + menu_data + "Translation: " + objBody.message.result.translatedText);
                    //res.end({Store:store_data, Menu:menu_data, Translation:objBody.message.result.translatedText});
                    res.end(jsondata);
                  } else {
                    res.status(response.statusCode).end();
                    console.log('error = ' + response.statusCode);
                  }
                });

              });
              //res.send("결과: \n" + store_data);
            });
          //  res.end("결과: \n"+data);
        });
    }
});

app.listen(3000, function () {
    console.log("Working on port 3000");
});
