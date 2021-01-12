const mongoose = require('mongoose');

var Schema = mongoose.Schema;
var store = new Schema({
    _id: Schema.Types.ObjectId,
    Store_Name: String,
    Store_Addr: String,
    Store_Tel: String,
    Store_Cate: String
});

module.exports = mongoose.model('store', store, 'store');
