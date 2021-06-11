/********* 判断是否支持webp图片 ***************/
var isSupportWebp = function () {
    try {
        return document.createElement('canvas').toDataURL('image/webp', 0.5).indexOf('data:image/webp') === 0;
    } catch (err) {
        return false;
    }
}
isSupportWebp()

/********* 带参数回调 ***************/
//jsons.callback = 'monitor'
window[jsons.callback].call(window, jsons.renderUrl)

/********* jquery 遍历每个元素 *********/
$(".webp-jpg").each(function () {
    othis = $(this);
    if (isSupportWebp()) {
        othis.attr("src", othis.attr("webp-src"));
    } else {
        othis.attr("src", othis.attr("jpg-src"));
    }
});

/********* 判断图片是否正常加载 ***************/
$(".webp-jpg").each(function(){
    othis = $(this);
    var img = new Image(); //创建一个Image对象，实现图片的预下载
    img.src = othis.attr("src");
    if (img.complete) { // 如果图片已经存在于浏览器缓存，直接调用回调函数
        return; // 直接返回，不用再处理onload事件
    }
    othis.on('load', function() {
        console.log("加载完成！");
       //图片加载完毕
    }).on('error',function (data) {
        console.log("加载失败！");
        console.log(data);
        othis2 = $(data.target)
        othis2.attr("src",othis2.attr("jpg-src"));
    })
});