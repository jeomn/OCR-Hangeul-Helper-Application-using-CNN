const mongoose = require('mongoose');

var Schema = mongoose.Schema;
var menu = new Schema({
    _id: Schema.Types.ObjectId,
    Menu_Cate: String,
    Menu_ko_Name: String,
    Menu_en_Name: String,
    Menu_Info: String
});

module.exports = mongoose.model('menu', menu, 'menu');
