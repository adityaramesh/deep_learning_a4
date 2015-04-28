let SessionLoad = 1
if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <S-Tab> <Plug>SuperTabBackward
inoremap <silent> <C-Tab> =UltiSnips#ListSnippets()
inoremap <Right> <Nop>
inoremap <Left> <Nop>
inoremap <Down> <Nop>
inoremap <Up> <Nop>
snoremap <silent>  c
xnoremap <silent> 	 :call UltiSnips#SaveLastVisualSelection()gvs
snoremap <silent> 	 :call UltiSnips#ExpandSnippetOrJump()
nnoremap <silent>  :nohl
snoremap  "_c
vnoremap < <gv
vnoremap > >gv
xnoremap <silent> <expr> A targets#e('A')
onoremap <silent> Ala :call targets#o('alA')
onoremap <silent> Ana :call targets#o('anA')
onoremap <silent> Aa :call targets#o('acA')
onoremap <silent> AL$ :call targets#o('$LA')
onoremap <silent> AN$ :call targets#o('$NA')
onoremap <silent> Al$ :call targets#o('$lA')
onoremap <silent> An$ :call targets#o('$nA')
onoremap <silent> A$ :call targets#o('$cA')
onoremap <silent> AL& :call targets#o('&LA')
onoremap <silent> AN& :call targets#o('&NA')
onoremap <silent> Al& :call targets#o('&lA')
onoremap <silent> An& :call targets#o('&nA')
onoremap <silent> A& :call targets#o('&cA')
onoremap <silent> AL| :call targets#o('|LA')
onoremap <silent> AN| :call targets#o('|NA')
onoremap <silent> Al| :call targets#o('|lA')
onoremap <silent> An| :call targets#o('|nA')
onoremap <silent> A| :call targets#o('|cA')
onoremap <silent> AL\ :call targets#o('\LA')
onoremap <silent> AN\ :call targets#o('\NA')
onoremap <silent> Al\ :call targets#o('\lA')
onoremap <silent> An\ :call targets#o('\nA')
onoremap <silent> A\ :call targets#o('\cA')
onoremap <silent> AL/ :call targets#o('/LA')
onoremap <silent> AN/ :call targets#o('/NA')
onoremap <silent> Al/ :call targets#o('/lA')
onoremap <silent> An/ :call targets#o('/nA')
onoremap <silent> A/ :call targets#o('/cA')
onoremap <silent> AL# :call targets#o('#LA')
onoremap <silent> AN# :call targets#o('#NA')
onoremap <silent> Al# :call targets#o('#lA')
onoremap <silent> An# :call targets#o('#nA')
onoremap <silent> A# :call targets#o('#cA')
onoremap <silent> AL* :call targets#o('*LA')
onoremap <silent> AN* :call targets#o('*NA')
onoremap <silent> Al* :call targets#o('*lA')
onoremap <silent> An* :call targets#o('*nA')
onoremap <silent> A* :call targets#o('*cA')
onoremap <silent> AL_ :call targets#o('_LA')
onoremap <silent> AN_ :call targets#o('_NA')
onoremap <silent> Al_ :call targets#o('_lA')
onoremap <silent> An_ :call targets#o('_nA')
onoremap <silent> A_ :call targets#o('_cA')
onoremap <silent> AL~ :call targets#o('~LA')
onoremap <silent> AN~ :call targets#o('~NA')
onoremap <silent> Al~ :call targets#o('~lA')
onoremap <silent> An~ :call targets#o('~nA')
onoremap <silent> A~ :call targets#o('~cA')
onoremap <silent> AL= :call targets#o('=LA')
onoremap <silent> AN= :call targets#o('=NA')
onoremap <silent> Al= :call targets#o('=lA')
onoremap <silent> An= :call targets#o('=nA')
onoremap <silent> A= :call targets#o('=cA')
onoremap <silent> AL- :call targets#o('-LA')
onoremap <silent> AN- :call targets#o('-NA')
onoremap <silent> Al- :call targets#o('-lA')
onoremap <silent> An- :call targets#o('-nA')
onoremap <silent> A- :call targets#o('-cA')
onoremap <silent> AL+ :call targets#o('+LA')
onoremap <silent> AN+ :call targets#o('+NA')
onoremap <silent> Al+ :call targets#o('+lA')
onoremap <silent> An+ :call targets#o('+nA')
onoremap <silent> A+ :call targets#o('+cA')
onoremap <silent> AL: :call targets#o(':LA')
onoremap <silent> AN: :call targets#o(':NA')
onoremap <silent> Al: :call targets#o(':lA')
onoremap <silent> An: :call targets#o(':nA')
onoremap <silent> A: :call targets#o(':cA')
onoremap <silent> AL; :call targets#o(';LA')
onoremap <silent> AN; :call targets#o(';NA')
onoremap <silent> Al; :call targets#o(';lA')
onoremap <silent> An; :call targets#o(';nA')
onoremap <silent> A; :call targets#o(';cA')
onoremap <silent> AL. :call targets#o('.LA')
onoremap <silent> AN. :call targets#o('.NA')
onoremap <silent> Al. :call targets#o('.lA')
onoremap <silent> An. :call targets#o('.nA')
onoremap <silent> A. :call targets#o('.cA')
onoremap <silent> AL, :call targets#o(',LA')
onoremap <silent> AN, :call targets#o(',NA')
onoremap <silent> Al, :call targets#o(',lA')
onoremap <silent> An, :call targets#o(',nA')
onoremap <silent> A, :call targets#o(',cA')
onoremap <silent> AL` :call targets#o('`LA')
onoremap <silent> AN` :call targets#o('`NA')
onoremap <silent> Al` :call targets#o('`lA')
onoremap <silent> An` :call targets#o('`nA')
onoremap <silent> A` :call targets#o('`cA')
onoremap <silent> AL' :call targets#o('''LA')
onoremap <silent> AN' :call targets#o('''NA')
onoremap <silent> Al' :call targets#o('''lA')
onoremap <silent> An' :call targets#o('''nA')
onoremap <silent> A' :call targets#o('''cA')
onoremap <silent> AL" :call targets#o('"LA')
onoremap <silent> AN" :call targets#o('"NA')
onoremap <silent> Al" :call targets#o('"lA')
onoremap <silent> An" :call targets#o('"nA')
onoremap <silent> A" :call targets#o('"cA')
onoremap <silent> Alt :call targets#o('tlA')
onoremap <silent> Ant :call targets#o('tnA')
onoremap <silent> At :call targets#o('tcA')
onoremap <silent> Al> :call targets#o('>lA')
onoremap <silent> An> :call targets#o('>nA')
onoremap <silent> A> :call targets#o('>cA')
onoremap <silent> Al< :call targets#o('<lA')
onoremap <silent> An< :call targets#o('<nA')
onoremap <silent> A< :call targets#o('<cA')
onoremap <silent> Al] :call targets#o(']lA')
onoremap <silent> An] :call targets#o(']nA')
onoremap <silent> A] :call targets#o(']cA')
onoremap <silent> Al[ :call targets#o('[lA')
onoremap <silent> An[ :call targets#o('[nA')
onoremap <silent> A[ :call targets#o('[cA')
onoremap <silent> AlB :call targets#o('BlA')
onoremap <silent> AnB :call targets#o('BnA')
onoremap <silent> AB :call targets#o('BcA')
onoremap <silent> Al} :call targets#o('}lA')
onoremap <silent> An} :call targets#o('}nA')
onoremap <silent> A} :call targets#o('}cA')
onoremap <silent> Al{ :call targets#o('{lA')
onoremap <silent> An{ :call targets#o('{nA')
onoremap <silent> A{ :call targets#o('{cA')
onoremap <silent> Alb :call targets#o('blA')
onoremap <silent> Anb :call targets#o('bnA')
onoremap <silent> Ab :call targets#o('bcA')
onoremap <silent> Al) :call targets#o(')lA')
onoremap <silent> An) :call targets#o(')nA')
onoremap <silent> A) :call targets#o(')cA')
onoremap <silent> Al( :call targets#o('(lA')
onoremap <silent> An( :call targets#o('(nA')
onoremap <silent> A( :call targets#o('(cA')
xnoremap <silent> <expr> I targets#e('I')
onoremap <silent> Ila :call targets#o('alI')
onoremap <silent> Ina :call targets#o('anI')
onoremap <silent> Ia :call targets#o('acI')
onoremap <silent> IL$ :call targets#o('$LI')
onoremap <silent> IN$ :call targets#o('$NI')
onoremap <silent> Il$ :call targets#o('$lI')
onoremap <silent> In$ :call targets#o('$nI')
onoremap <silent> I$ :call targets#o('$cI')
onoremap <silent> IL& :call targets#o('&LI')
onoremap <silent> IN& :call targets#o('&NI')
onoremap <silent> Il& :call targets#o('&lI')
onoremap <silent> In& :call targets#o('&nI')
onoremap <silent> I& :call targets#o('&cI')
onoremap <silent> IL| :call targets#o('|LI')
onoremap <silent> IN| :call targets#o('|NI')
onoremap <silent> Il| :call targets#o('|lI')
onoremap <silent> In| :call targets#o('|nI')
onoremap <silent> I| :call targets#o('|cI')
onoremap <silent> IL\ :call targets#o('\LI')
onoremap <silent> IN\ :call targets#o('\NI')
onoremap <silent> Il\ :call targets#o('\lI')
onoremap <silent> In\ :call targets#o('\nI')
onoremap <silent> I\ :call targets#o('\cI')
onoremap <silent> IL/ :call targets#o('/LI')
onoremap <silent> IN/ :call targets#o('/NI')
onoremap <silent> Il/ :call targets#o('/lI')
onoremap <silent> In/ :call targets#o('/nI')
onoremap <silent> I/ :call targets#o('/cI')
onoremap <silent> IL# :call targets#o('#LI')
onoremap <silent> IN# :call targets#o('#NI')
onoremap <silent> Il# :call targets#o('#lI')
onoremap <silent> In# :call targets#o('#nI')
onoremap <silent> I# :call targets#o('#cI')
onoremap <silent> IL* :call targets#o('*LI')
onoremap <silent> IN* :call targets#o('*NI')
onoremap <silent> Il* :call targets#o('*lI')
onoremap <silent> In* :call targets#o('*nI')
onoremap <silent> I* :call targets#o('*cI')
onoremap <silent> IL_ :call targets#o('_LI')
onoremap <silent> IN_ :call targets#o('_NI')
onoremap <silent> Il_ :call targets#o('_lI')
onoremap <silent> In_ :call targets#o('_nI')
onoremap <silent> I_ :call targets#o('_cI')
onoremap <silent> IL~ :call targets#o('~LI')
onoremap <silent> IN~ :call targets#o('~NI')
onoremap <silent> Il~ :call targets#o('~lI')
onoremap <silent> In~ :call targets#o('~nI')
onoremap <silent> I~ :call targets#o('~cI')
onoremap <silent> IL= :call targets#o('=LI')
onoremap <silent> IN= :call targets#o('=NI')
onoremap <silent> Il= :call targets#o('=lI')
onoremap <silent> In= :call targets#o('=nI')
onoremap <silent> I= :call targets#o('=cI')
onoremap <silent> IL- :call targets#o('-LI')
onoremap <silent> IN- :call targets#o('-NI')
onoremap <silent> Il- :call targets#o('-lI')
onoremap <silent> In- :call targets#o('-nI')
onoremap <silent> I- :call targets#o('-cI')
onoremap <silent> IL+ :call targets#o('+LI')
onoremap <silent> IN+ :call targets#o('+NI')
onoremap <silent> Il+ :call targets#o('+lI')
onoremap <silent> In+ :call targets#o('+nI')
onoremap <silent> I+ :call targets#o('+cI')
onoremap <silent> IL: :call targets#o(':LI')
onoremap <silent> IN: :call targets#o(':NI')
onoremap <silent> Il: :call targets#o(':lI')
onoremap <silent> In: :call targets#o(':nI')
onoremap <silent> I: :call targets#o(':cI')
onoremap <silent> IL; :call targets#o(';LI')
onoremap <silent> IN; :call targets#o(';NI')
onoremap <silent> Il; :call targets#o(';lI')
onoremap <silent> In; :call targets#o(';nI')
onoremap <silent> I; :call targets#o(';cI')
onoremap <silent> IL. :call targets#o('.LI')
onoremap <silent> IN. :call targets#o('.NI')
onoremap <silent> Il. :call targets#o('.lI')
onoremap <silent> In. :call targets#o('.nI')
onoremap <silent> I. :call targets#o('.cI')
onoremap <silent> IL, :call targets#o(',LI')
onoremap <silent> IN, :call targets#o(',NI')
onoremap <silent> Il, :call targets#o(',lI')
onoremap <silent> In, :call targets#o(',nI')
onoremap <silent> I, :call targets#o(',cI')
onoremap <silent> IL` :call targets#o('`LI')
onoremap <silent> IN` :call targets#o('`NI')
onoremap <silent> Il` :call targets#o('`lI')
onoremap <silent> In` :call targets#o('`nI')
onoremap <silent> I` :call targets#o('`cI')
onoremap <silent> IL' :call targets#o('''LI')
onoremap <silent> IN' :call targets#o('''NI')
onoremap <silent> Il' :call targets#o('''lI')
onoremap <silent> In' :call targets#o('''nI')
onoremap <silent> I' :call targets#o('''cI')
onoremap <silent> IL" :call targets#o('"LI')
onoremap <silent> IN" :call targets#o('"NI')
onoremap <silent> Il" :call targets#o('"lI')
onoremap <silent> In" :call targets#o('"nI')
onoremap <silent> I" :call targets#o('"cI')
onoremap <silent> Ilt :call targets#o('tlI')
onoremap <silent> Int :call targets#o('tnI')
onoremap <silent> It :call targets#o('tcI')
onoremap <silent> Il> :call targets#o('>lI')
onoremap <silent> In> :call targets#o('>nI')
onoremap <silent> I> :call targets#o('>cI')
onoremap <silent> Il< :call targets#o('<lI')
onoremap <silent> In< :call targets#o('<nI')
onoremap <silent> I< :call targets#o('<cI')
onoremap <silent> Il] :call targets#o(']lI')
onoremap <silent> In] :call targets#o(']nI')
onoremap <silent> I] :call targets#o(']cI')
onoremap <silent> Il[ :call targets#o('[lI')
onoremap <silent> In[ :call targets#o('[nI')
onoremap <silent> I[ :call targets#o('[cI')
onoremap <silent> IlB :call targets#o('BlI')
onoremap <silent> InB :call targets#o('BnI')
onoremap <silent> IB :call targets#o('BcI')
onoremap <silent> Il} :call targets#o('}lI')
onoremap <silent> In} :call targets#o('}nI')
onoremap <silent> I} :call targets#o('}cI')
onoremap <silent> Il{ :call targets#o('{lI')
onoremap <silent> In{ :call targets#o('{nI')
onoremap <silent> I{ :call targets#o('{cI')
onoremap <silent> Ilb :call targets#o('blI')
onoremap <silent> Inb :call targets#o('bnI')
onoremap <silent> Ib :call targets#o('bcI')
onoremap <silent> Il) :call targets#o(')lI')
onoremap <silent> In) :call targets#o(')nI')
onoremap <silent> I) :call targets#o(')cI')
onoremap <silent> Il( :call targets#o('(lI')
onoremap <silent> In( :call targets#o('(nI')
onoremap <silent> I( :call targets#o('(cI')
xmap S <Plug>VSurround
nmap \\u <Plug>CommentaryUndo:echomsg '\\ is deprecated. Use gc'
nmap \\\ <Plug>CommentaryLine:echomsg '\\ is deprecated. Use gc'
nmap \\ :echomsg '\\ is deprecated. Use gc'<Plug>Commentary
xmap \\ <Plug>Commentary:echomsg '\\ is deprecated. Use gc'
xnoremap <silent> <expr> a targets#e('a')
onoremap <silent> ala :call targets#o('ala')
onoremap <silent> ana :call targets#o('ana')
onoremap <silent> aa :call targets#o('aca')
onoremap <silent> aL$ :call targets#o('$La')
onoremap <silent> aN$ :call targets#o('$Na')
onoremap <silent> al$ :call targets#o('$la')
onoremap <silent> an$ :call targets#o('$na')
onoremap <silent> a$ :call targets#o('$ca')
onoremap <silent> aL& :call targets#o('&La')
onoremap <silent> aN& :call targets#o('&Na')
onoremap <silent> al& :call targets#o('&la')
onoremap <silent> an& :call targets#o('&na')
onoremap <silent> a& :call targets#o('&ca')
onoremap <silent> aL| :call targets#o('|La')
onoremap <silent> aN| :call targets#o('|Na')
onoremap <silent> al| :call targets#o('|la')
onoremap <silent> an| :call targets#o('|na')
onoremap <silent> a| :call targets#o('|ca')
onoremap <silent> aL\ :call targets#o('\La')
onoremap <silent> aN\ :call targets#o('\Na')
onoremap <silent> al\ :call targets#o('\la')
onoremap <silent> an\ :call targets#o('\na')
onoremap <silent> a\ :call targets#o('\ca')
onoremap <silent> aL/ :call targets#o('/La')
onoremap <silent> aN/ :call targets#o('/Na')
onoremap <silent> al/ :call targets#o('/la')
onoremap <silent> an/ :call targets#o('/na')
onoremap <silent> a/ :call targets#o('/ca')
onoremap <silent> aL# :call targets#o('#La')
onoremap <silent> aN# :call targets#o('#Na')
onoremap <silent> al# :call targets#o('#la')
onoremap <silent> an# :call targets#o('#na')
onoremap <silent> a# :call targets#o('#ca')
onoremap <silent> aL* :call targets#o('*La')
onoremap <silent> aN* :call targets#o('*Na')
onoremap <silent> al* :call targets#o('*la')
onoremap <silent> an* :call targets#o('*na')
onoremap <silent> a* :call targets#o('*ca')
onoremap <silent> aL_ :call targets#o('_La')
onoremap <silent> aN_ :call targets#o('_Na')
onoremap <silent> al_ :call targets#o('_la')
onoremap <silent> an_ :call targets#o('_na')
onoremap <silent> a_ :call targets#o('_ca')
onoremap <silent> aL~ :call targets#o('~La')
onoremap <silent> aN~ :call targets#o('~Na')
onoremap <silent> al~ :call targets#o('~la')
onoremap <silent> an~ :call targets#o('~na')
onoremap <silent> a~ :call targets#o('~ca')
onoremap <silent> aL= :call targets#o('=La')
onoremap <silent> aN= :call targets#o('=Na')
onoremap <silent> al= :call targets#o('=la')
onoremap <silent> an= :call targets#o('=na')
onoremap <silent> a= :call targets#o('=ca')
onoremap <silent> aL- :call targets#o('-La')
onoremap <silent> aN- :call targets#o('-Na')
onoremap <silent> al- :call targets#o('-la')
onoremap <silent> an- :call targets#o('-na')
onoremap <silent> a- :call targets#o('-ca')
onoremap <silent> aL+ :call targets#o('+La')
onoremap <silent> aN+ :call targets#o('+Na')
onoremap <silent> al+ :call targets#o('+la')
onoremap <silent> an+ :call targets#o('+na')
onoremap <silent> a+ :call targets#o('+ca')
onoremap <silent> aL: :call targets#o(':La')
onoremap <silent> aN: :call targets#o(':Na')
onoremap <silent> al: :call targets#o(':la')
onoremap <silent> an: :call targets#o(':na')
onoremap <silent> a: :call targets#o(':ca')
onoremap <silent> aL; :call targets#o(';La')
onoremap <silent> aN; :call targets#o(';Na')
onoremap <silent> al; :call targets#o(';la')
onoremap <silent> an; :call targets#o(';na')
onoremap <silent> a; :call targets#o(';ca')
onoremap <silent> aL. :call targets#o('.La')
onoremap <silent> aN. :call targets#o('.Na')
onoremap <silent> al. :call targets#o('.la')
onoremap <silent> an. :call targets#o('.na')
onoremap <silent> a. :call targets#o('.ca')
onoremap <silent> aL, :call targets#o(',La')
onoremap <silent> aN, :call targets#o(',Na')
onoremap <silent> al, :call targets#o(',la')
onoremap <silent> an, :call targets#o(',na')
onoremap <silent> a, :call targets#o(',ca')
onoremap <silent> aL` :call targets#o('`La')
onoremap <silent> aN` :call targets#o('`Na')
onoremap <silent> al` :call targets#o('`la')
onoremap <silent> an` :call targets#o('`na')
onoremap <silent> a` :call targets#o('`ca')
onoremap <silent> aL' :call targets#o('''La')
onoremap <silent> aN' :call targets#o('''Na')
onoremap <silent> al' :call targets#o('''la')
onoremap <silent> an' :call targets#o('''na')
onoremap <silent> a' :call targets#o('''ca')
onoremap <silent> aL" :call targets#o('"La')
onoremap <silent> aN" :call targets#o('"Na')
onoremap <silent> al" :call targets#o('"la')
onoremap <silent> an" :call targets#o('"na')
onoremap <silent> a" :call targets#o('"ca')
onoremap <silent> alt :call targets#o('tla')
onoremap <silent> ant :call targets#o('tna')
onoremap <silent> at :call targets#o('tca')
onoremap <silent> al> :call targets#o('>la')
onoremap <silent> an> :call targets#o('>na')
onoremap <silent> a> :call targets#o('>ca')
onoremap <silent> al< :call targets#o('<la')
onoremap <silent> an< :call targets#o('<na')
onoremap <silent> a< :call targets#o('<ca')
onoremap <silent> al] :call targets#o(']la')
onoremap <silent> an] :call targets#o(']na')
onoremap <silent> a] :call targets#o(']ca')
onoremap <silent> al[ :call targets#o('[la')
onoremap <silent> an[ :call targets#o('[na')
onoremap <silent> a[ :call targets#o('[ca')
onoremap <silent> alB :call targets#o('Bla')
onoremap <silent> anB :call targets#o('Bna')
onoremap <silent> aB :call targets#o('Bca')
onoremap <silent> al} :call targets#o('}la')
onoremap <silent> an} :call targets#o('}na')
onoremap <silent> a} :call targets#o('}ca')
onoremap <silent> al{ :call targets#o('{la')
onoremap <silent> an{ :call targets#o('{na')
onoremap <silent> a{ :call targets#o('{ca')
onoremap <silent> alb :call targets#o('bla')
onoremap <silent> anb :call targets#o('bna')
onoremap <silent> ab :call targets#o('bca')
onoremap <silent> al) :call targets#o(')la')
onoremap <silent> an) :call targets#o(')na')
onoremap <silent> a) :call targets#o(')ca')
onoremap <silent> al( :call targets#o('(la')
onoremap <silent> an( :call targets#o('(na')
onoremap <silent> a( :call targets#o('(ca')
nmap cgc <Plug>ChangeCommentary
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
vmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
nmap gcu <Plug>Commentary<Plug>Commentary
nmap gcc <Plug>CommentaryLine
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
xmap gS <Plug>VgSurround
nnoremap gj j
nnoremap gk k
xnoremap <silent> <expr> i targets#e('i')
onoremap <silent> ila :call targets#o('ali')
onoremap <silent> ina :call targets#o('ani')
onoremap <silent> ia :call targets#o('aci')
onoremap <silent> iL$ :call targets#o('$Li')
onoremap <silent> iN$ :call targets#o('$Ni')
onoremap <silent> il$ :call targets#o('$li')
onoremap <silent> in$ :call targets#o('$ni')
onoremap <silent> i$ :call targets#o('$ci')
onoremap <silent> iL& :call targets#o('&Li')
onoremap <silent> iN& :call targets#o('&Ni')
onoremap <silent> il& :call targets#o('&li')
onoremap <silent> in& :call targets#o('&ni')
onoremap <silent> i& :call targets#o('&ci')
onoremap <silent> iL| :call targets#o('|Li')
onoremap <silent> iN| :call targets#o('|Ni')
onoremap <silent> il| :call targets#o('|li')
onoremap <silent> in| :call targets#o('|ni')
onoremap <silent> i| :call targets#o('|ci')
onoremap <silent> iL\ :call targets#o('\Li')
onoremap <silent> iN\ :call targets#o('\Ni')
onoremap <silent> il\ :call targets#o('\li')
onoremap <silent> in\ :call targets#o('\ni')
onoremap <silent> i\ :call targets#o('\ci')
onoremap <silent> iL/ :call targets#o('/Li')
onoremap <silent> iN/ :call targets#o('/Ni')
onoremap <silent> il/ :call targets#o('/li')
onoremap <silent> in/ :call targets#o('/ni')
onoremap <silent> i/ :call targets#o('/ci')
onoremap <silent> iL# :call targets#o('#Li')
onoremap <silent> iN# :call targets#o('#Ni')
onoremap <silent> il# :call targets#o('#li')
onoremap <silent> in# :call targets#o('#ni')
onoremap <silent> i# :call targets#o('#ci')
onoremap <silent> iL* :call targets#o('*Li')
onoremap <silent> iN* :call targets#o('*Ni')
onoremap <silent> il* :call targets#o('*li')
onoremap <silent> in* :call targets#o('*ni')
onoremap <silent> i* :call targets#o('*ci')
onoremap <silent> iL_ :call targets#o('_Li')
onoremap <silent> iN_ :call targets#o('_Ni')
onoremap <silent> il_ :call targets#o('_li')
onoremap <silent> in_ :call targets#o('_ni')
onoremap <silent> i_ :call targets#o('_ci')
onoremap <silent> iL~ :call targets#o('~Li')
onoremap <silent> iN~ :call targets#o('~Ni')
onoremap <silent> il~ :call targets#o('~li')
onoremap <silent> in~ :call targets#o('~ni')
onoremap <silent> i~ :call targets#o('~ci')
onoremap <silent> iL= :call targets#o('=Li')
onoremap <silent> iN= :call targets#o('=Ni')
onoremap <silent> il= :call targets#o('=li')
onoremap <silent> in= :call targets#o('=ni')
onoremap <silent> i= :call targets#o('=ci')
onoremap <silent> iL- :call targets#o('-Li')
onoremap <silent> iN- :call targets#o('-Ni')
onoremap <silent> il- :call targets#o('-li')
onoremap <silent> in- :call targets#o('-ni')
onoremap <silent> i- :call targets#o('-ci')
onoremap <silent> iL+ :call targets#o('+Li')
onoremap <silent> iN+ :call targets#o('+Ni')
onoremap <silent> il+ :call targets#o('+li')
onoremap <silent> in+ :call targets#o('+ni')
onoremap <silent> i+ :call targets#o('+ci')
onoremap <silent> iL: :call targets#o(':Li')
onoremap <silent> iN: :call targets#o(':Ni')
onoremap <silent> il: :call targets#o(':li')
onoremap <silent> in: :call targets#o(':ni')
onoremap <silent> i: :call targets#o(':ci')
onoremap <silent> iL; :call targets#o(';Li')
onoremap <silent> iN; :call targets#o(';Ni')
onoremap <silent> il; :call targets#o(';li')
onoremap <silent> in; :call targets#o(';ni')
onoremap <silent> i; :call targets#o(';ci')
onoremap <silent> iL. :call targets#o('.Li')
onoremap <silent> iN. :call targets#o('.Ni')
onoremap <silent> il. :call targets#o('.li')
onoremap <silent> in. :call targets#o('.ni')
onoremap <silent> i. :call targets#o('.ci')
onoremap <silent> iL, :call targets#o(',Li')
onoremap <silent> iN, :call targets#o(',Ni')
onoremap <silent> il, :call targets#o(',li')
onoremap <silent> in, :call targets#o(',ni')
onoremap <silent> i, :call targets#o(',ci')
onoremap <silent> iL` :call targets#o('`Li')
onoremap <silent> iN` :call targets#o('`Ni')
onoremap <silent> il` :call targets#o('`li')
onoremap <silent> in` :call targets#o('`ni')
onoremap <silent> i` :call targets#o('`ci')
onoremap <silent> iL' :call targets#o('''Li')
onoremap <silent> iN' :call targets#o('''Ni')
onoremap <silent> il' :call targets#o('''li')
onoremap <silent> in' :call targets#o('''ni')
onoremap <silent> i' :call targets#o('''ci')
onoremap <silent> iL" :call targets#o('"Li')
onoremap <silent> iN" :call targets#o('"Ni')
onoremap <silent> il" :call targets#o('"li')
onoremap <silent> in" :call targets#o('"ni')
onoremap <silent> i" :call targets#o('"ci')
onoremap <silent> ilt :call targets#o('tli')
onoremap <silent> int :call targets#o('tni')
onoremap <silent> it :call targets#o('tci')
onoremap <silent> il> :call targets#o('>li')
onoremap <silent> in> :call targets#o('>ni')
onoremap <silent> i> :call targets#o('>ci')
onoremap <silent> il< :call targets#o('<li')
onoremap <silent> in< :call targets#o('<ni')
onoremap <silent> i< :call targets#o('<ci')
onoremap <silent> il] :call targets#o(']li')
onoremap <silent> in] :call targets#o(']ni')
onoremap <silent> i] :call targets#o(']ci')
onoremap <silent> il[ :call targets#o('[li')
onoremap <silent> in[ :call targets#o('[ni')
onoremap <silent> i[ :call targets#o('[ci')
onoremap <silent> ilB :call targets#o('Bli')
onoremap <silent> inB :call targets#o('Bni')
onoremap <silent> iB :call targets#o('Bci')
onoremap <silent> il} :call targets#o('}li')
onoremap <silent> in} :call targets#o('}ni')
onoremap <silent> i} :call targets#o('}ci')
onoremap <silent> il{ :call targets#o('{li')
onoremap <silent> in{ :call targets#o('{ni')
onoremap <silent> i{ :call targets#o('{ci')
onoremap <silent> ilb :call targets#o('bli')
onoremap <silent> inb :call targets#o('bni')
onoremap <silent> ib :call targets#o('bci')
onoremap <silent> il) :call targets#o(')li')
onoremap <silent> in) :call targets#o(')ni')
onoremap <silent> i) :call targets#o(')ci')
onoremap <silent> il( :call targets#o('(li')
onoremap <silent> in( :call targets#o('(ni')
onoremap <silent> i( :call targets#o('(ci')
nnoremap j gj
nnoremap k gk
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
vnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(expand((exists("g:netrw_gx")? g:netrw_gx : '<cfile>')),netrw#CheckIfRemote())
nmap <silent> <Plug>CommentaryUndo <Plug>Commentary<Plug>Commentary
nnoremap <silent> <Plug>SurroundRepeat .
snoremap <silent> <Del> c
snoremap <silent> <BS> c
snoremap <silent> <C-Tab> :call UltiSnips#ListSnippets()
imap S <Plug>ISurround
imap s <Plug>Isurround
inoremap <silent> 	 =UltiSnips#ExpandSnippetOrJump()
imap  <Plug>Isurround
cabbr rename =getcmdpos() == 1 && getcmdtype() == ":" ? "Rename" : "rename"
cnoreabbr <expr> help getcmdtype() == ":" && getcmdline() == "help" ? "vertical help" : "help"
cnoreabbr <expr> h getcmdtype() == ":" && getcmdline() == "h" ? "vertical h" : "h"
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set backspace=indent,eol,start
set completeopt=menu,menuone
set fileencodings=ucs-bom,utf-8,default,latin1
set formatoptions=tcqor2
set helplang=en
set hidden
set incsearch
set pumheight=20
set ruler
set runtimepath=~/.vim,~/.vim/bundle/Vundle.vim,~/.vim/bundle/tabular,~/.vim/bundle/smartfile,~/.vim/bundle/rename.vim,~/.vim/bundle/vim-colorschemes,~/.vim/bundle/vim-colors-solarized,~/.vim/bundle/clang_complete,~/.vim/bundle/supertab,~/.vim/bundle/ultisnips,~/.vim/bundle/vim-snippets,~/.vim/bundle/vim-markdown,~/.vim/bundle/vim-surround,~/.vim/bundle/targets.vim,~/.vim/bundle/vim-repeat,~/.vim/bundle/vim-commentary,/usr/share/vim/vimfiles,/usr/share/vim/vim74,/usr/share/vim/vimfiles/after,~/.vim/after,~/.vim/bundle/Vundle.vim,~/.vim/bundle/Vundle.vim/after,~/.vim/bundle/tabular/after,~/.vim/bundle/smartfile/after,~/.vim/bundle/rename.vim/after,~/.vim/bundle/vim-colorschemes/after,~/.vim/bundle/vim-colors-solarized/after,~/.vim/bundle/clang_complete/after,~/.vim/bundle/supertab/after,~/.vim/bundle/ultisnips/after,~/.vim/bundle/vim-snippets/after,~/.vim/bundle/vim-markdown/after,~/.vim/bundle/vim-surround/after,~/.vim/bundle/targets.vim/after,~/.vim/bundle/vim-repeat/after,~/.vim/bundle/vim-commentary/after
set scrolloff=999
set sessionoptions=blank,buffers,folds,help,options,tabpages,winsize,sesdir
set showcmd
set showmatch
set splitright
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc,.png,.jpg
set textwidth=80
set ttimeoutlen=100
set visualbell
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
exe "cd " . escape(expand("<sfile>:p:h"), ' ')
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +0 nngraph_handin.lua
argglobal
silent! argdel *
argadd ~/deep_learning/submissions/assignment_4/nngraph/nngraph_handin.lua
edit nngraph_handin.lua
set splitbelow splitright
set nosplitbelow
wincmd t
set winheight=1 winwidth=1
argglobal
setlocal keymap=
setlocal noarabic
setlocal autoindent
setlocal backupcopy=
setlocal nobinary
setlocal nobreakindent
setlocal breakindentopt=
setlocal bufhidden=
setlocal buflisted
setlocal buftype=
setlocal nocindent
setlocal cinkeys=0{,0},0),:,0#,!^F,o,O,e
setlocal cinoptions=
setlocal cinwords=if,else,while,do,for,switch
setlocal colorcolumn=
setlocal comments=:--
setlocal commentstring=--%s
setlocal complete=.,w,b,u,t,i
set concealcursor=vin
setlocal concealcursor=vin
set conceallevel=2
setlocal conceallevel=2
setlocal completefunc=
setlocal nocopyindent
setlocal cryptmethod=
setlocal nocursorbind
setlocal nocursorcolumn
set cursorline
setlocal cursorline
setlocal define=
setlocal dictionary=
setlocal nodiff
setlocal equalprg=
setlocal errorformat=
setlocal noexpandtab
if &filetype != 'lua'
setlocal filetype=lua
endif
setlocal foldcolumn=0
setlocal foldenable
setlocal foldexpr=0
setlocal foldignore=#
setlocal foldlevel=0
setlocal foldmarker={{{,}}}
setlocal foldmethod=manual
setlocal foldminlines=1
setlocal foldnestmax=20
setlocal foldtext=foldtext()
setlocal formatexpr=
setlocal formatoptions=2croql
setlocal formatlistpat=^\\s*\\d\\+[\\]:.)}\\t\ ]\\s*
setlocal grepprg=
setlocal iminsert=0
setlocal imsearch=0
setlocal include=
setlocal includeexpr=
setlocal indentexpr=
setlocal indentkeys=0{,0},:,0#,!^F,o,O,e
setlocal noinfercase
setlocal iskeyword=@,48-57,_,192-255
setlocal keywordprg=
setlocal nolinebreak
setlocal nolisp
setlocal lispwords=
setlocal nolist
setlocal makeprg=
setlocal matchpairs=(:),{:},[:]
setlocal modeline
setlocal modifiable
setlocal nrformats=octal,hex
setlocal nonumber
setlocal numberwidth=4
setlocal omnifunc=
setlocal path=
setlocal nopreserveindent
setlocal nopreviewwindow
setlocal quoteescape=\\
setlocal noreadonly
setlocal norelativenumber
setlocal norightleft
setlocal rightleftcmd=search
setlocal noscrollbind
setlocal shiftwidth=8
setlocal noshortname
setlocal nosmartindent
setlocal softtabstop=0
setlocal nospell
setlocal spellcapcheck=[.?!]\\_[\\])'\"\	\ ]\\+
setlocal spellfile=
setlocal spelllang=en
setlocal statusline=
setlocal suffixesadd=.lua
setlocal swapfile
setlocal synmaxcol=3000
if &syntax != 'lua'
setlocal syntax=lua
endif
setlocal tabstop=8
setlocal tags=
setlocal textwidth=80
setlocal thesaurus=
setlocal noundofile
setlocal undolevels=-123456
setlocal nowinfixheight
setlocal nowinfixwidth
setlocal wrap
setlocal wrapmargin=0
silent! normal! zE
let s:l = 26 - ((25 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
26
normal! 0
tabnext 1
if exists('s:wipebuf')
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 shortmess=filnxtToO
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
