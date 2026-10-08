
var myArgs = process.argv.slice(2);
console.log('myArgs: ', myArgs);
var str1 = 'proj/eb/searchimg/'
var str2 = str1.concat(myArgs[1])
var str3 = str2.concat('r.png')

const puppeteer = require('puppeteer')
var request = require('request');
var headers = {
    'apikey': process.env.ZENSCRAPE_API_KEY  // key removed from source; it was committed earlier and should be revoked
};        
    
var options = { 
    url: myArgs[0],
    headers: headers
};

function callback(error, response, body) {
    if (!error && response.statusCode == 200) {
        var html = body;
    }
    puppeteer.launch({ headless: true }).then(async browser => {
        console.log('Running tests..')
        const page = await browser.newPage()
        await page.setContent(html)
        await page.waitForTimeout(0)
        await page.screenshot({ path: str3, fullPage: true })
        await browser.close()
    })


}

request(options, callback);

