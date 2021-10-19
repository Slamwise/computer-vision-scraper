// puppeteer-extra is a drop-in replacement for puppeteer,
// it augments the installed puppeteer with plugin functionality

var myArgs = process.argv.slice(2);
console.log('myArgs: ', myArgs);
var str1 = 'proj/eb/searchimg/'
var str2 = str1.concat(myArgs[1])
var str3 = str2.concat('r.png')
var str4 = myArgs[3]
var port = parseInt(str4, 10)

const puppeteer = require('puppeteer-extra')

// add stealth plugin and use defaults (all evasion techniques)
const StealthPlugin = require('puppeteer-extra-plugin-stealth')


const pluginProxy = require('puppeteer-extra-plugin-proxy')
 puppeteer.use(StealthPlugin(), pluginProxy({
   address: myArgs[2],
   port: 	myArgs[3]
 }))

const randomUseragent = require('random-useragent')
const userAgent = randomUseragent.getRandom(function (ua) {
  return ua.browserName === 'Firefox' && parseFloat(ua.browserVersion) >= 80;
})
const UA = userAgent
const UserAgentOverride = require('puppeteer-extra-plugin-stealth/evasions/user-agent-override')
const ua = UserAgentOverride({
  userAgent: UA,
  locale: 'en-US,en'
})
puppeteer.use(ua)



// puppeteer usage as normal
puppeteer.launch({ headless: true }).then(async browser => {
  console.log('Running tests..')
  const page = await browser.newPage()
  await page.goto(myArgs[0])
  await page.waitForTimeout(5000)
  await page.screenshot({ path: str3, fullPage: true })
  await browser.close()
})