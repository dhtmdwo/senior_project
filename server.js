const express = require('express')
const session = require('express-session')
const path = require('path');
const app = express()
const port = 3001

const db = require('./lib/db');
const sessionOption = require('./lib/sessionOption');
const bodyParser = require("body-parser");
const bcrypt = require('bcrypt');

app.use(express.static(path.join(__dirname, '/build')));
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

var MySQLStore = require('express-mysql-session')(session);
var sessionStore = new MySQLStore(sessionOption);
app.use(session({  
	key: 'session_cookie_name',
    secret: '~',
	store: sessionStore,
	resave: false,
	saveUninitialized: false
}))

app.get('/', (req, res) => {    
    req.sendFile(path.join(__dirname, '/build/index.html'));
})

app.get('/authcheck', (req, res) => {      
    const sendData = { isLogin: "" };
    if (req.session.is_logined) {
        sendData.isLogin = "True"
    } else {
        sendData.isLogin = "False"
    }
    res.send(sendData);
})

app.get('/logout', function (req, res) {
    req.session.destroy(function (err) {
        res.redirect('/');
    });
});

app.post("/login", (req, res) => { // ������ �޾Ƽ� ��� ����
    const username = req.body.userId;
    const password = req.body.userPassword;
    const sendData = { isLogin: "" };

    if (username && password) {             // id�� pw�� �ԷµǾ����� Ȯ��
        db.query('SELECT * FROM userTable WHERE username = ?', [username], function (error, results, fields) {
            if (error) throw error;
            if (results.length > 0) {       // db������ ��ȯ���� �ִ� = ��ġ�ϴ� ���̵� �ִ�.      
                if (password === results[0].password) {                  // ��й�ȣ�� ��ġ�ϸ�
                    req.session.is_logined = true;      // ���� ���� ����
                    req.session.nickname = username;
                    req.session.save(function () {
                        sendData.isLogin = "True"
                        res.send(sendData);
                    });
                    db.query(`INSERT INTO logTable (created, username, action, command, actiondetail) VALUES (NOW(), ?, 'login' , ?, ?)`
                        , [req.session.nickname, '-', `React login`], function (error, result) { });
                }
                else{                                   // ��й�ȣ�� �ٸ� ���
                    sendData.isLogin = "Incorrect PW!"
                    res.send(sendData);
                }
                                      
            } else {    // db�� �ش� ���̵� ���� ���
                sendData.isLogin = "Incorrect ID!"
                res.send(sendData);
            }
        });
    } else {            // ���̵�, ��й�ȣ �� �Էµ��� ���� ���� �ִ� ���
        sendData.isLogin = "Please fill in the ID and PW!"
        res.send(sendData);
    }
});

app.post("/signin", (req, res) => {  // ������ �޾Ƽ� ��� ����
    const username = req.body.userId;
    const password = req.body.userPassword;
    const password2 = req.body.userPassword2;
    
    const sendData = { isSuccess: "" };

    if (username && password && password2) {
        db.query('SELECT * FROM userTable WHERE username = ?', [username], function(error, results, fields) { // DB�� ���� �̸��� ȸ�����̵� �ִ��� Ȯ��
            if (error) throw error;
            if (results.length <= 0 && password == password2) {         // DB�� ���� �̸��� ȸ�����̵� ����, ��й�ȣ�� �ùٸ��� �Էµ� ��� ��
                db.query('INSERT INTO userTable (username, password) VALUES(?,?)', [username, password], function (error, data) {
                    if (error) throw error;
                    req.session.save(function () {                        
                        sendData.isSuccess = "True"
                        res.send(sendData);
                    });
                });
            } else if (password != password2) {                     // ��й�ȣ�� �ùٸ��� �Էµ��� ���� ���                  
                sendData.isSuccess = "PW already exists!"
                res.send(sendData);
            }
            else {                                                  // DB�� ���� �̸��� ȸ�����̵� �ִ� ���            
                sendData.isSuccess = "ID already exists!"
                res.send(sendData);  
            }            
        });        
    } else {
        sendData.isSuccess = "Please fill in the ID and PW!"
        res.send(sendData);  
    }
    
});


app.listen(port, () => {
    console.log(`Example app listening at http://localhost:${port}`)
})