export default async function handler(req, res) {
    if (req.method !== 'POST') return res.status(405).end();
    const { email } = req.body;
    if (!email) return res.status(400).json({ error: 'Email required' });
  
    await fetch('https://api.resend.com/audiences/YOUR_AUDIENCE_ID/contacts', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.RESEND_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ email, unsubscribed: false }),
    });
  
    return res.status(200).json({ ok: true });
  }