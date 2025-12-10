const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  const backendProxy = createProxyMiddleware({
    target: 'https://10.20.104.250:8443',
    changeOrigin: true,
    secure: false, // 允许自签名证书
  });

  app.use('/search', backendProxy);
  app.use('/images', backendProxy);
  app.use('/health', backendProxy);
};

