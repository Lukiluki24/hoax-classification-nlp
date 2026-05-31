chrome.storage.local.get(['isShieldActive'], function(result) {
  if (result.isShieldActive === false) {
    return;
  }

  (async function autoScanAllHeadings() {
    const headingElements = document.querySelectorAll('h1, h2, h3, h4');
    if (headingElements.length === 0) return;

    const blacklistKeywords = [
      "baca juga", "berita terpopuler", "komentar", "artikel terkait", 
      "trending", "top news", "indeks berita", "rekomendasi",
      "pilihan editor", "jangan lewatkan", "berlangganan", "tag populer",
      "semua artikel", "informasi berita", "kategori", "penulis",
      "detikcom", "detiknews", "detiksport", "detikjabar",
      "tag terpopuler", "video terpopuler", "ke halaman video"
    ];

    headingElements.forEach((heading) => {
      if (heading.querySelector('.indobert-auto-badge')) return;
      if (heading.closest('footer, nav, aside, header, .sidebar, .widget, .menu, .header, .breadcrumb, .pagination')) return;

      let headline = heading.innerText.trim();
      if (!headline) return;

      // --- FILTER 1: Buang Hashtag ---
      if (headline.startsWith('#')) return;

      // --- FILTER 2: Minimum Kata ---
      const validWords = headline.split(/\s+/).filter(word => word.length > 1);
      
      if (validWords.length < 5) return; 

      // --- FILTER 3: Blacklist ---
      const lowerHeadline = headline.toLowerCase();
      const isBlacklisted = blacklistKeywords.some(keyword => lowerHeadline.includes(keyword));
      if (isBlacklisted) return;

      // =========================================================================
      // TESTING PURPOSE: Bersihkan tag label di awal judul (contoh: [SALAH], [FAKTA])
      headline = headline.replace(/^\[.*?\]\s*/, '').trim(); 
      // =========================================================================

      if (headline.split(/\s+/).filter(w => w.length > 1).length < 4) return;

      chrome.runtime.sendMessage({ action: "checkHoax", headline: headline }, (response) => {
        if (!response || !response.success) return;

        const data = response.data;
        if (heading.querySelector('.indobert-auto-badge')) return;

        const badge = document.createElement('span');
        badge.classList.add('indobert-auto-badge');

        if (data.prediction === "Hoax") {
          badge.textContent = "LIKELY TO BE HOAX";
          badge.classList.add('indobert-hoax');
        } else {
          badge.textContent = "LIKELY TO BE FACT";
          badge.classList.add('indobert-faktual');
        }

        heading.appendChild(badge);
      });
    });
  })();
});