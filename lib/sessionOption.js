// MySQL�� ������ ������ �ɼ�
var options = {  
  host: 'localhost',
  user: 'root',
  password: 'rudehs97!',
  database: 'seniorproject',
  port: 3306,
  
  clearExpired : true ,             // ����� ���� �ڵ� Ȯ�� �� ����� ����
  checkExpirationInterval: 10000,   // ����� ������ �������� �� (milliseconds)
  expiration: 1000*60*60*2,         // ��ȿ�� ������ �ִ� �Ⱓ 2�ð����� ���� (milliseconds) 
};

module.exports = options;